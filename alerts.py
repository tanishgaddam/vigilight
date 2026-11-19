"""Alert dispatch. Every alert is stored with evidence (snapshot + short clip) and sent
to every channel you configured:

  dashboard  full-screen flashing overlay + siren + spoken warning + acknowledge button
  telegram   photo with caption, then the video clip
  discord    webhook message with snapshot, then the clip
  ntfy       phone push notification with the snapshot attached (ntfy.sh app)
  mqtt       JSON message for other systems (street light controller, police console, ...)

Only HIGH / CRITICAL alerts are pushed out; WARNING entries stay on the dashboard.
"""
import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

import config as cfg

log = logging.getLogger("vigilight.alerts")


def save_snapshot(camera, alert_type, frame):
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    name = f"{camera}_{stamp}_{alert_type}.jpg"
    cv2.imwrite(str(cfg.EVIDENCE_DIR / name), frame)
    return name


def write_clip(frames, stem, fps):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    for codec, ext in (("avc1", ".mp4"), ("mp4v", ".mp4")):  # H.264 plays everywhere; mp4v is the fallback
        name = stem + ext
        writer = cv2.VideoWriter(str(cfg.EVIDENCE_DIR / name), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if not writer.isOpened():
            continue
        for f in frames:
            writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        writer.release()
        return name
    return None


def format_text(a):
    lines = [
        f"🚨 {a['type'].replace('_', ' ')} ({a['severity']})",
        a["message"],
        f"Camera: {a['camera']} | Dispatch: {a['service']}",
        f"Time: {a['time']} | Confidence: {int(a['score'] * 100)}%",
    ]
    if a.get("reasons"):
        lines.append("Why: " + "; ".join(a["reasons"]))
    return "\n".join(lines)


class AlertDispatcher:
    def __init__(self):
        self.history = deque(maxlen=100)
        self._lock = threading.Lock()
        self._seq = 0
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._mqtt = None
        if cfg.MQTT_ENABLED:
            try:
                import paho.mqtt.client as mqtt

                self._mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                self._mqtt.connect_async(cfg.MQTT_BROKER, 1883, 60)
                self._mqtt.loop_start()
            except Exception as exc:
                log.warning("MQTT disabled: %s", exc)
                self._mqtt = None

    # --- public API ---------------------------------------------------------
    def channels(self):
        return {
            "dashboard": True,
            "telegram": bool(cfg.TELEGRAM_BOT_TOKEN and cfg.TELEGRAM_CHAT_ID),
            "discord": bool(cfg.DISCORD_WEBHOOK_URL),
            "ntfy": bool(cfg.NTFY_TOPIC),
            "mqtt": self._mqtt is not None,
        }

    def publish(self, alert):
        alert = dict(alert)
        with self._lock:
            self._seq += 1
            alert.update(id=self._seq, time=time.strftime("%H:%M:%S"), ts=time.time(), acked=False,
                         clip=alert.get("clip"))
            self.history.appendleft(alert)
        if alert["severity"] in ("HIGH", "CRITICAL"):
            self._pool.submit(self._notify, alert)
        return alert

    def attach_clip(self, alert_id, filename):
        with self._lock:
            alert = next((a for a in self.history if a["id"] == alert_id), None)
            if alert is None or not filename:
                return
            alert["clip"] = filename
        if alert["severity"] in ("HIGH", "CRITICAL"):
            self._pool.submit(self._send_clip, alert)

    def ack(self, alert_id):
        with self._lock:
            for a in self.history:
                if a["id"] == alert_id:
                    a["acked"] = True
                    return True
        return False

    def list(self):
        with self._lock:
            return [dict(a) for a in self.history]

    # --- channels -----------------------------------------------------------
    def _notify(self, alert):
        for send in (self._send_mqtt, self._send_telegram, self._send_discord, self._send_ntfy):
            try:
                send(alert)
            except Exception as exc:
                log.warning("%s failed: %s", send.__name__, exc)

    def _send_mqtt(self, a):
        if self._mqtt is None:
            return
        payload = {k: a[k] for k in ("type", "service", "severity", "camera", "message", "score", "reasons", "time")}
        self._mqtt.publish(cfg.MQTT_TOPIC, json.dumps(payload))

    def _send_telegram(self, a):
        if not (cfg.TELEGRAM_BOT_TOKEN and cfg.TELEGRAM_CHAT_ID):
            return
        base = f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/"
        path = cfg.EVIDENCE_DIR / a["snapshot"] if a.get("snapshot") else None
        if path and path.exists():
            with open(path, "rb") as f:
                requests.post(base + "sendPhoto", data={"chat_id": cfg.TELEGRAM_CHAT_ID, "caption": format_text(a)},
                              files={"photo": f}, timeout=20)
        else:
            requests.post(base + "sendMessage", json={"chat_id": cfg.TELEGRAM_CHAT_ID, "text": format_text(a)}, timeout=10)

    def _send_discord(self, a):
        if not cfg.DISCORD_WEBHOOK_URL:
            return
        path = cfg.EVIDENCE_DIR / a["snapshot"] if a.get("snapshot") else None
        body = {"content": format_text(a)}
        if path and path.exists():
            with open(path, "rb") as f:
                requests.post(cfg.DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(body)},
                              files={"file": (path.name, f, "image/jpeg")}, timeout=20)
        else:
            requests.post(cfg.DISCORD_WEBHOOK_URL, json=body, timeout=10)

    def _send_ntfy(self, a):
        if not cfg.NTFY_TOPIC:
            return
        ascii_text = f"{a['message']} | camera {a['camera']} | {int(a['score'] * 100)}%".encode("ascii", "ignore").decode()
        headers = {
            "Title": f"VigiLight {a['type']}",
            "Priority": "urgent" if a["severity"] == "CRITICAL" else "high",
            "Tags": "rotating_light",
            "Message": ascii_text,
        }
        url = f"{cfg.NTFY_SERVER}/{cfg.NTFY_TOPIC}"
        path = cfg.EVIDENCE_DIR / a["snapshot"] if a.get("snapshot") else None
        if path and path.exists():
            with open(path, "rb") as f:
                requests.put(url, data=f, headers={**headers, "Filename": path.name}, timeout=20)
        else:
            requests.post(url, data=ascii_text.encode(), headers=headers, timeout=10)

    def _send_clip(self, a):
        path = cfg.EVIDENCE_DIR / a["clip"]
        if not path.exists():
            return
        try:
            if cfg.TELEGRAM_BOT_TOKEN and cfg.TELEGRAM_CHAT_ID:
                with open(path, "rb") as f:
                    requests.post(f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendVideo",
                                  data={"chat_id": cfg.TELEGRAM_CHAT_ID, "caption": f"Clip: {a['type']} at {a['camera']}"},
                                  files={"video": f}, timeout=60)
        except Exception as exc:
            log.warning("telegram clip failed: %s", exc)
        try:
            if cfg.DISCORD_WEBHOOK_URL:
                with open(path, "rb") as f:
                    requests.post(cfg.DISCORD_WEBHOOK_URL, data={"content": f"Clip: {a['type']} at {a['camera']}"},
                                  files={"file": (path.name, f, "video/mp4")}, timeout=60)
        except Exception as exc:
            log.warning("discord clip failed: %s", exc)
