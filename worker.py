"""One CameraWorker per camera: reads frames, runs detection + behaviour analysis,
listens to audio, records evidence and raises alerts."""
import logging
import re
import threading
import time
from collections import deque

import cv2
import numpy as np

import alerts as alerts_mod
import config as cfg
from audio import AudioMonitor
from behavior import LEVEL_RANK, CollisionEngine, ThreatEngine
from sources import VideoSource

log = logging.getLogger("vigilight.worker")

LEVEL_COLORS = {"NORMAL": (80, 200, 80), "WATCH": (0, 215, 255), "WARNING": (0, 140, 255), "THREAT": (40, 40, 255)}


def _mask(source):
    return re.sub(r"//[^/@]+@", "//***@", str(source))


class CameraWorker:
    def __init__(self, name, source, audio, dispatcher):
        self.name = name
        self.source = source
        self.audio_cfg = audio
        self.dispatcher = dispatcher
        self.running = True
        self.video = VideoSource(source)
        self.threat = ThreatEngine()
        self.collision = CollisionEngine()
        self.detector = None
        self.loading = True
        self.error = None

        self.level, self.score, self.people, self.fps = "NORMAL", 0.0, 0, 0.0
        self.latest_view = None
        self._cond = threading.Condition()
        self._jpeg, self._seq = None, 0

        self._audio_alerts = deque(maxlen=10)
        self._audio_ctx = None            # (time, category, score)
        self._audio_text = "quiet"
        self._scream_hits = 0
        self._last_alert = {}
        self._buffer = deque(maxlen=max(1, int(cfg.PROCESS_FPS * cfg.CLIP_PRE_SEC)))
        self._pending_clips = []

        audio_src = cfg.derive_audio_source(source, audio)
        self.audio = AudioMonitor(name, audio_src, self.on_audio_scores) if (audio_src and cfg.AUDIO_ENABLED) else None
        self._thread = threading.Thread(target=self._run, daemon=True)

    # --- lifecycle ------------------------------------------------------------
    def start(self):
        self.video.start()
        self._thread.start()
        if self.audio:
            self.audio.start()

    def stop(self):
        self.running = False
        self.video.stop()
        if self.audio:
            self.audio.stop()
        with self._cond:
            self._cond.notify_all()

    # --- streaming --------------------------------------------------------------
    def _publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        with self._cond:
            self._jpeg, self.latest_view = buf.tobytes(), img
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq, timeout=2.0):
        with self._cond:
            if self._seq == last_seq and self.running:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    @staticmethod
    def _placeholder(text):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img

    # --- audio ------------------------------------------------------------------
    def on_audio_scores(self, scores):
        now = time.time()
        thr = cfg.AUDIO_THRESHOLD
        distress = {k: scores.get(k, 0.0) for k in ("GUNSHOT", "GLASS_BREAK", "SCREAM")}
        label = max(distress, key=distress.get)
        if distress[label] >= thr:
            self._audio_ctx = (now, label, distress[label])
        self._audio_text = f"{label.lower().replace('_', ' ')} {distress[label]:.2f}" if distress[label] >= 0.2 else "quiet"

        def push(kind, severity, message, score):
            self._audio_alerts.append({"type": kind, "service": "POLICE", "severity": severity, "message": message,
                                       "score": round(score, 2), "reasons": [f"audio classifier: {kind.lower()} {score:.2f}"]})

        if scores.get("GUNSHOT", 0) >= thr:
            push("GUNSHOT_AUDIO", "CRITICAL", "Gunshot-like sound detected", scores["GUNSHOT"])
        elif scores.get("GLASS_BREAK", 0) >= thr:
            push("GLASS_BREAK", "HIGH", "Glass breaking detected", scores["GLASS_BREAK"])
        self._scream_hits = self._scream_hits + 1 if scores.get("SCREAM", 0) >= thr else 0
        if self._scream_hits >= cfg.SCREAM_HITS:
            self._scream_hits = 0
            push("DISTRESS_SCREAM", "HIGH", "Screaming or shouting detected", scores["SCREAM"])
        if scores.get("LOUD_NOISE", 0) >= 0.95:  # loudness fallback: log only, never notifies
            push("LOUD_NOISE", "WARNING", "Very loud noise (audio model unavailable)", scores["LOUD_NOISE"])

    def _audio_context(self, now):
        if self._audio_ctx and now - self._audio_ctx[0] <= cfg.AUDIO_CONTEXT_SEC:
            return {"label": self._audio_ctx[1], "score": self._audio_ctx[2]}
        return None

    # --- main loop ----------------------------------------------------------------
    def _run(self):
        self._publish(self._placeholder(f"{self.name}: LOADING MODELS..."))
        try:
            from detector import Detector

            self.detector = Detector()
        except Exception as exc:
            self.error = f"model load failed: {exc}"[:160]
            self.loading = False
            log.error("[%s] %s", self.name, self.error)
            self._publish(self._placeholder("MODEL LOAD FAILED"))
            return
        self.loading = False
        interval = 1.0 / cfg.PROCESS_FPS
        last_id, last_offline = -1, 0.0
        fps_t, fps_n = time.time(), 0
        while self.running:
            t0 = time.time()
            frame, fid = self.video.get()
            if frame is None:
                if t0 - last_offline > 1.0:
                    self._publish(self._placeholder(f"{self.name}: OFFLINE"))
                    last_offline = t0
                time.sleep(0.1)
                continue
            if fid == last_id:
                time.sleep(0.02)
                continue
            last_id = fid
            try:
                self._process(frame)
                self.error = None
            except Exception as exc:
                self.error = f"processing error: {exc}"[:160]
                log.exception("[%s] processing error", self.name)
                time.sleep(0.5)
            fps_n += 1
            if time.time() - fps_t >= 2.0:
                self.fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            spare = interval - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)

    def _process(self, frame):
        now = time.time()
        det = self.detector.detect(frame)
        results, alerts = self.threat.update(det["persons"], det["weapons"], self._audio_context(now), now)
        alerts += self.collision.update(det["vehicles"], now)
        self.people = len(results)
        self.level = max((r["level"] for r in results), key=LEVEL_RANK.get, default="NORMAL")
        self.score = max((r["score"] for r in results), default=0.0)

        view = self._draw(frame, results, det)
        self._publish(view)
        self._buffer.append(view)
        for clip in list(self._pending_clips):
            clip["frames"].append(view)
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                self._pending_clips.remove(clip)
                threading.Thread(target=self._finish_clip, args=(clip,), daemon=True).start()

        while self._audio_alerts:
            alerts.append(self._audio_alerts.popleft())
        for a in alerts:
            self._raise(a, view, now)

    def _draw(self, frame, results, det):
        img = frame.copy()
        th = max(2, img.shape[1] // 450)
        scale = max(0.5, img.shape[1] / 1280)
        for r in results:
            color = LEVEL_COLORS[r["level"]]
            x1, y1, x2, y2 = map(int, r["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
            cv2.putText(img, f"#{r['id']} {r['level']} {r['score']:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)
            if r["kpts"] is not None:
                for idx in (9, 10):
                    x, y, c = r["kpts"][idx]
                    if c > 0.35:
                        cv2.circle(img, (int(x), int(y)), 4 * th, color, -1)
        for w in det["weapons"]:
            x1, y1, x2, y2 = map(int, w["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), th)
            cv2.putText(img, f"{w['label']} {w['conf']:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), th)
        for v in det["vehicles"]:
            x1, y1, x2, y2 = map(int, v["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(img, f"{v['label']} #{v['id']}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (255, 200, 0), 1)
        h, w = img.shape[:2]
        if w > cfg.STREAM_WIDTH:
            img = cv2.resize(img, (cfg.STREAM_WIDTH, int(h * cfg.STREAM_WIDTH / w)))
        hud = f"{self.name} | CAM {self.video.fps:.0f} fps | AI {self.fps:.0f} fps | {self.level}"
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEVEL_COLORS[self.level], 1)
        return img

    # --- alerts + evidence ----------------------------------------------------------
    def _raise(self, alert, view, now):
        if now - self._last_alert.get(alert["type"], 0) < cfg.ALERT_COOLDOWN:
            return
        self._last_alert[alert["type"]] = now
        alert = dict(alert, camera=self.name)
        alert["message"] = f"{alert['message']} at {self.name}"
        alert["snapshot"] = alerts_mod.save_snapshot(self.name, alert["type"], view)
        stored = self.dispatcher.publish(alert)
        if cfg.SAVE_CLIPS and alert["severity"] in ("HIGH", "CRITICAL"):
            self._pending_clips.append({
                "alert_id": stored["id"], "stem": alert["snapshot"].rsplit(".", 1)[0],
                "frames": list(self._buffer), "remaining": max(1, int(cfg.PROCESS_FPS * cfg.CLIP_POST_SEC)),
            })

    def _finish_clip(self, clip):
        try:
            name = alerts_mod.write_clip(clip["frames"], clip["stem"], cfg.PROCESS_FPS)
            self.dispatcher.attach_clip(clip["alert_id"], name)
        except Exception:
            log.exception("[%s] clip write failed", self.name)

    # --- status for the dashboard -------------------------------------------------------
    def status(self):
        audio = {"enabled": self.audio is not None, "backend": self.audio.backend if self.audio else None,
                 "error": self.audio.error if self.audio else None, "last": self._audio_text}
        return {"name": self.name, "source": _mask(self.source), "kind": "usb" if isinstance(self.source, int) else "network",
                "online": self.video.online, "loading": self.loading, "fps": round(self.fps, 1), "cam_fps": round(self.video.fps, 1),
                "level": self.level, "score": round(self.score, 2), "people": self.people,
                "audio": audio, "error": self.error}
