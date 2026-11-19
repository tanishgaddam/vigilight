"""Audio AI. Google's YAMNet (521 AudioSet sound classes) listens to each camera's
microphone and reports glass breaking, screaming/shouting and gunshots.

Audio sources per camera:
  "auto"   IP Webcam style cameras: http://host:port/audio.wav next to the /video URL
  "mic"    default local microphone (good partner for a USB camera)
  "mic:2"  local microphone with a specific device index
  <url>    any http WAV stream

If TensorFlow is not installed the monitor falls back to a plain loudness check and
reports it as low-severity LOUD_NOISE only.
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
                except Exception as exc:  # TensorFlow missing, offline, ...
                    log.warning("YAMNet unavailable (%s) - using loudness fallback", exc)
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
        self.backend = "yamnet" if clf else "loudness"
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
        if clf is not None:
            scores = clf.classify(window)
        else:
            rms = float(np.sqrt(np.mean(window ** 2)))
            scores = {"LOUD_NOISE": min(1.0, rms / 0.3)}
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
