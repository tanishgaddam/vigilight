"""Audio AI. Google's YAMNet (521 AudioSet classes) or built-in Physics-Based
Spectral Acoustic Classifier listens to each camera's microphone and reports:
  - GLASS_BREAK (high-frequency shattering & brittle fracture transients)
  - SCREAM (distress vocal resonance & elevated harmonic energy)
  - GUNSHOT (explosive high-crest acoustic impulse)

Audio sources per camera:
  "auto"   IP Webcam style cameras: http://host:port/audio.wav next to the /video URL
  "mic"    default local microphone (good partner for a USB camera)
  "mic:2"  local microphone with a specific device index
  <url>    any http WAV stream
"""
import logging
import queue
import struct
import threading
import time
from math import gcd

import numpy as np
import requests
from scipy.signal import resample_poly

log = logging.getLogger("vigilight.audio")

TARGET_SR = 16000
WINDOW_SEC = 2.0
HOP_SEC = 1.0

CATEGORIES = {
    "GLASS_BREAK": ["Glass", "Shatter", "Breaking", "Smash, crash"],
    "GUNSHOT": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"],
    "SCREAM": ["Screaming", "Shout", "Yell", "Bellow"],
}


class YamnetClassifier:
    _instance = None
    _tried = False
    _guard = threading.Lock()

    @classmethod
    def get(cls):
        """Load YAMNet once, shared by all cameras. Returns None if unavailable."""
        with cls._guard:
            if not cls._tried:
                cls._tried = True
                try:
                    cls._instance = cls()
                    log.info("YAMNet loaded")
                except Exception as exc:  # TensorFlow missing, pkg_resources missing, offline...
                    log.warning("YAMNet unavailable (%s) - using spectral classifier fallback", exc)
                    cls._instance = None
            return cls._instance

    def __init__(self):
        import csv
        import tensorflow_hub as hub

        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        path = self._model.class_map_path().numpy().decode("utf-8")
        with open(path, newline="", encoding="utf-8") as f:
            names = [row["display_name"] for row in csv.DictReader(f)]
        index = {n: i for i, n in enumerate(names)}
        self._cat_idx = {cat: [index[n] for n in labels if n in index] for cat, labels in CATEGORIES.items()}
        self._lock = threading.Lock()

    def classify(self, wave):
        with self._lock:
            scores, _, _ = self._model(wave.astype(np.float32))
        peak = scores.numpy().max(axis=0)  # max over frames: glass breaks are short
        return {cat: float(peak[idx].max()) if idx else 0.0 for cat, idx in self._cat_idx.items()}


class SpectralAcousticClassifier:
    """Offline spectral analysis acoustic classifier for Glass Break, Scream, and Bang/Gunshot.
    Uses FFT frequency distribution, zero-crossing rate, and transient crest factor.
    Runs 100% locally with zero extra dependencies.
    """

    def classify(self, wave, sr=TARGET_SR):
        if len(wave) == 0:
            return {"GLASS_BREAK": 0.0, "SCREAM": 0.0, "GUNSHOT": 0.0, "LOUD_NOISE": 0.0}

        rms = float(np.sqrt(np.mean(wave ** 2)))
        peak = float(np.max(np.abs(wave)))
        crest = peak / (rms + 1e-6)

        # Baseline noise gate
        if rms < 0.012:
            return {"GLASS_BREAK": 0.0, "SCREAM": 0.0, "GUNSHOT": 0.0, "LOUD_NOISE": 0.0}

        n = len(wave)
        fft_vals = np.abs(np.fft.rfft(wave))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        total_energy = float(np.sum(fft_vals ** 2) + 1e-8)

        # Acoustic Spectral Bands:
        # Low: 50 - 800 Hz (thuds, rumbles, base speech)
        # Mid: 800 - 3000 Hz (vocal resonance, distress cries)
        # High: 2800 - 8000 Hz (glass fracture, shatter, metallic clinks)
        low_band = (freqs >= 50) & (freqs < 800)
        mid_band = (freqs >= 800) & (freqs < 3000)
        high_band = (freqs >= 2800) & (freqs <= 8000)

        e_low = float(np.sum(fft_vals[low_band] ** 2) / total_energy)
        e_mid = float(np.sum(fft_vals[mid_band] ** 2) / total_energy)
        e_high = float(np.sum(fft_vals[high_band] ** 2) / total_energy)

        # Zero-Crossing Rate (ZCR) — high for noisy/shattering frequencies
        zcr = float(np.mean(np.abs(np.diff(np.sign(wave)))) / 2.0)

        # 1. Glass Break detection:
        # High energy in the 2.8k-8kHz band, elevated ZCR, and sudden transient burst
        glass_score = 0.0
        if e_high > 0.18 or (e_high > 0.12 and zcr > 0.11):
            vol_boost = min(1.0, rms / 0.035)
            glass_score = min(1.0, (e_high * 1.7 + zcr * 1.5) * vol_boost)

        # 2. Scream detection:
        # Human scream: dominant mid-frequency vocal resonance (800-3000Hz)
        scream_score = 0.0
        if e_mid > 0.35 and rms > 0.025:
            scream_score = min(1.0, (e_mid * 1.5) * min(1.0, rms / 0.07))

        # 3. Gunshot / Loud Bang:
        # High crest factor (explosive transient spike)
        gun_score = 0.0
        if peak > 0.35 and crest > 4.0 and rms > 0.035:
            gun_score = min(1.0, (crest / 7.0) * min(1.0, peak / 0.6))

        loud_score = min(1.0, rms / 0.25)

        return {
            "GLASS_BREAK": float(np.clip(glass_score, 0.0, 1.0)),
            "SCREAM": float(np.clip(scream_score, 0.0, 1.0)),
            "GUNSHOT": float(np.clip(gun_score, 0.0, 1.0)),
            "LOUD_NOISE": float(np.clip(loud_score, 0.0, 1.0)),
        }


def _resample(x, sr):
    if sr == TARGET_SR:
        return x.astype(np.float32)
    g = gcd(int(sr), TARGET_SR)
    return resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)


class AudioMonitor:
    def __init__(self, cam_name, source, on_scores):
        self.cam_name = cam_name
        self.source = source
        self.on_scores = on_scores
        self.running = False
        self.backend = None
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.running = True
        self._thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        clf = YamnetClassifier.get()
        if clf is None:
            clf = SpectralAcousticClassifier()
            self.backend = "spectral"
        else:
            self.backend = "yamnet"

        while self.running:
            try:
                if self.source.startswith("mic"):
                    device = int(self.source.split(":", 1)[1]) if ":" in self.source else None
                    self._run_mic(device, clf)
                else:
                    self._run_http(self.source, clf)
            except Exception as exc:
                self.error = str(exc)[:120]
                log.warning("[%s] audio error: %s", self.cam_name, exc)
            if self.running:
                time.sleep(3)

    def _handle(self, window, clf):
        scores = clf.classify(window)
        self.error = None
        self.on_scores(scores)

    def _run_http(self, url, clf):
        with requests.get(url, stream=True, timeout=(5, 10)) as r:
            r.raise_for_status()
            header = bytearray()
            sr, ch, got_header = 44100, 1, False
            pending = b""
            buf = np.zeros(0, np.float32)
            for chunk in r.iter_content(chunk_size=4096):
                if not self.running:
                    return
                if not chunk:
                    continue
                if not got_header:
                    header += chunk
                    if len(header) < 44:
                        continue
                    if header[:4] == b"RIFF":
                        ch = struct.unpack("<H", header[22:24])[0] or 1
                        sr = struct.unpack("<I", header[24:28])[0] or 44100
                        chunk = bytes(header[44:])
                    else:
                        chunk = bytes(header)
                    got_header = True
                pending += chunk
                usable = len(pending) - (len(pending) % (2 * ch))
                if usable <= 0:
                    continue
                data = np.frombuffer(pending[:usable], dtype="<i2").astype(np.float32) / 32768.0
                pending = pending[usable:]
                if ch > 1:
                    data = data.reshape(-1, ch).mean(axis=1)
                buf = np.concatenate([buf, data])
                win_n, hop_n = int(sr * WINDOW_SEC), int(sr * HOP_SEC)
                while len(buf) >= win_n:
                    self._handle(_resample(buf[:win_n], sr), clf)
                    buf = buf[hop_n:]

    def _run_mic(self, device, clf):
        import sounddevice as sd

        q = queue.Queue(maxsize=200)

        def callback(indata, frames, t, status):
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        sr = TARGET_SR
        try:
            stream = sd.InputStream(device=device, channels=1, samplerate=sr, callback=callback)
        except Exception:
            sr = int(sd.query_devices(device, "input")["default_samplerate"])
            stream = sd.InputStream(device=device, channels=1, samplerate=sr, callback=callback)
        buf = np.zeros(0, np.float32)
        win_n, hop_n = int(sr * WINDOW_SEC), int(sr * HOP_SEC)
        with stream:
            while self.running:
                try:
                    buf = np.concatenate([buf, q.get(timeout=1)])
                except queue.Empty:
                    continue
                while len(buf) >= win_n:
                    self._handle(_resample(buf[:win_n], sr), clf)
                    buf = buf[hop_n:]
