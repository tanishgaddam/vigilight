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
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._mqtt = None
        if cfg.MQTT_ENABLED:
            def _connect_mqtt():
                try:
                    import paho.mqtt.client as mqtt

                    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                    client.connect(cfg.MQTT_BROKER, 1883, 60)
                    client.loop_start()
                    self._mqtt = client
                except Exception as exc:
                    log.warning("MQTT disabled: %s", exc)
                    self._mqtt = None

            threading.Thread(target=_connect_mqtt, daemon=True).start()

    # --- public API ---------------------------------------------------------
    def channels(self):
        ch = {"dashboard": True}
        if cfg.TELEGRAM_ENABLED:
            ch["telegram"] = bool(cfg.TELEGRAM_BOT_TOKEN and cfg.TELEGRAM_CHAT_ID)
        if cfg.DISCORD_ENABLED:
            ch["discord"] = bool(cfg.DISCORD_WEBHOOK_URL)
        ch["ntfy"] = bool(cfg.NTFY_TOPIC)
        ch["mqtt"] = self._mqtt is not None
        return ch

    def publish(self, alert):
        alert = dict(alert)
        demo = bool(alert.get("demo"))
        high = alert["severity"] in ("HIGH", "CRITICAL")
        notifiable = high and (not demo or cfg.DEMO_NOTIFY)
        hold = notifiable and cfg.SETTINGS["verify_first"]
        with self._lock:
            self._seq += 1
            alert.update(id=self._seq, time=time.strftime("%H:%M:%S"), ts=time.time(), acked=False,
                         clip=alert.get("clip"), status="pending" if hold else "new", responder=None,
                         responded_in=None, escalated=False, notifiable=notifiable,
                         confidence=cfg.confidence_label(alert["score"]))
            self.history.appendleft(alert)
        if hold:
            timer = threading.Timer(cfg.VERIFY_TIMEOUT, self._timeout, args=(alert["id"],))
            timer.daemon = True
            timer.start()
        elif notifiable:
            self._escalate(alert)
        return alert

    def _find(self, alert_id):
        return next((a for a in self.history if a["id"] == alert_id), None)

    def _escalate(self, alert):
        """Send the alert (and its clip, if already recorded) to the external channels, once."""
        with self._lock:
            if alert["escalated"] or not alert["notifiable"]:
                return
            alert["escalated"] = True
            has_clip = bool(alert.get("clip"))
        log.info("Escalating alert #%s (%s, %s) to external channels", alert["id"], alert["type"], alert["severity"])
        self._notify(alert)
        if has_clip:
            self._pool.submit(self._send_clip, alert)

    def _timeout(self, alert_id):
        with self._lock:
            alert = self._find(alert_id)
            if alert is None or alert["status"] != "pending":
                return
            alert["status"] = "new"
            alert["auto_escalated"] = True
        self._escalate(alert)

    def attach_clip(self, alert_id, filename):
        with self._lock:
            alert = self._find(alert_id)
            if alert is None or not filename:
                return
            alert["clip"] = filename
            send = alert["escalated"]
        if send:
            self._pool.submit(self._send_clip, alert)

    def act(self, alert_id, action, name=None):
        """Guard actions: confirm (real), dismiss (false alarm), respond (on my way), resolve (done), ack."""
        with self._lock:
            alert = self._find(alert_id)
            if alert is None:
                return None
            if action == "confirm":
                alert["status"] = "confirmed"
                if name:
                    alert["responder"] = name[:30]
            elif action == "dismiss":
                alert["status"], alert["acked"] = "dismissed", True
                if name:
                    alert["responder"] = name[:30]
            elif action == "respond":
                alert["status"], alert["acked"] = "responding", True
                alert["responder"] = (name or "Guard")[:30]
                if alert.get("responded_in") is None:
                    alert["responded_in"] = round(time.time() - alert["ts"], 1)
            elif action == "resolve":
                alert["status"], alert["acked"] = "resolved", True
                if name:
                    alert["responder"] = name[:30]
            elif action == "ack":
                alert["status"], alert["acked"] = "closed", True
                if name:
                    alert["responder"] = name[:30]
                elif not alert.get("responder"):
                    alert["responder"] = "Command Center"
            else:
                return None
            alert["handled_at"] = time.strftime("%H:%M:%S")
        if action in ("confirm", "respond"):
            self._escalate(alert)   # a guard says it is real: release it to the other channels
        return dict(alert)

    def ack(self, alert_id, name=None):
        return self.act(alert_id, "ack", name=name) is not None

    def stats(self):
        with self._lock:
            items = [a for a in self.history if a["severity"] != "WARNING" and a["type"] != "TEST_ALERT"]
        real = [a for a in items if a["status"] in ("confirmed", "responding", "resolved")]
        false = [a for a in items if a["status"] == "dismissed"]
        times = [a["responded_in"] for a in items if a["responded_in"] is not None]
        reviewed = len(real) + len(false)
        return {"total": len(items), "confirmed": len(real), "false_alarms": len(false),
                "waiting": len([a for a in items if a["status"] in ("new", "pending")]),
                "precision": round(len(real) / reviewed, 2) if reviewed else None,
                "avg_response_sec": round(sum(times) / len(times), 1) if times else None}

    def list(self):
        with self._lock:
            return [dict(a) for a in self.history]

    # --- channels -----------------------------------------------------------
    def _notify(self, alert):
        """Every channel runs on its own worker thread, so a slow or unreachable channel
        can never delay the ntfy push."""
        for send in (self._send_ntfy, self._send_mqtt, self._send_telegram, self._send_discord):
            self._pool.submit(self._safe_send, send, alert)

    @staticmethod
    def _safe_send(send, alert):
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
                resp = requests.put(url, data=f, headers={**headers, "Filename": path.name}, timeout=20)
        else:
            resp = requests.post(url, data=ascii_text.encode(), headers=headers, timeout=10)
        if resp.ok:
            log.info("ntfy push sent for alert #%s", a["id"])
        else:
            log.warning("ntfy rejected alert #%s: HTTP %s %s", a["id"], resp.status_code, resp.text[:120])

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
