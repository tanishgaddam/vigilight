"""VigiLight - AI street-light surveillance. Run:  python app.py  then open http://localhost:5000

Cameras can be added in the dashboard, or on the command line:
    python app.py --camera 0 --camera http://192.168.1.20:8080/video --camera rtsp://user:pw@host/stream
"""
import argparse
import hashlib
import hmac
import io
import logging
import secrets
import socket
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_from_directory

import alerts as alerts_mod
import config as cfg
from alerts import AlertDispatcher
from demo import SCENARIOS, create_demo_workers
from sources import scan_usb
from worker import CameraWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", static_url_path="/static")
dispatcher = AlertDispatcher()
workers = {}
lock = threading.RLock()


def persist():
    cfg.save_camera_configs([{"name": w.name, "source": w.source, "audio": w.audio_cfg}
                             for w in workers.values() if not w.is_demo])


def add_camera(name, source, audio=None, save=True):
    src = cfg.normalize_source(source)
    with lock:
        if any(w.source == src for w in workers.values()):
            raise ValueError("That camera is already added.")
        base = cfg.safe_name(name or (f"usb_{src}" if isinstance(src, int) else "camera"))
        name, n = base, 2
        while name in workers:
            name, n = f"{base}_{n}", n + 1
        if audio is None:
            audio = "auto" if isinstance(src, str) and src.startswith("http") else "off"
        worker = CameraWorker(name, src, audio, dispatcher)
        workers[name] = worker
        worker.start()
        if save:
            persist()
    return worker


# --- optional access PIN (set ACCESS_PIN in .env; recommended whenever guards use phones on Wi-Fi) ---
_SECRET = secrets.token_hex(16)


def _auth_token():
    return hashlib.sha256((cfg.ACCESS_PIN + _SECRET).encode()).hexdigest()


@app.before_request
def require_pin():
    if not cfg.ACCESS_PIN or request.path in ("/login", "/api/login", "/static/login.html", "/static/manifest.json", "/static/icon.svg"):
        return None
    if hmac.compare_digest(request.cookies.get("vl_auth", ""), _auth_token()):
        return None
    if request.path.startswith(("/api/", "/video_feed", "/evidence", "/guard_qr")):
        return jsonify({"error": "login required"}), 401
    return redirect("/login?next=" + ("/guard" if request.path == "/guard" else "/"))


@app.route("/login")
def login_page():
    return app.send_static_file("login.html")


@app.route("/api/login", methods=["POST"])
def login():
    pin = str((request.get_json(silent=True) or {}).get("pin", ""))
    if not cfg.ACCESS_PIN or hmac.compare_digest(pin, cfg.ACCESS_PIN):
        resp = jsonify({"ok": True})
        if cfg.ACCESS_PIN:
            resp.set_cookie("vl_auth", _auth_token(), httponly=True, samesite="Lax", max_age=60 * 60 * 24)
        return resp
    time.sleep(1.0)  # slows down PIN guessing
    return jsonify({"error": "Wrong PIN"}), 401


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the right network card
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def guard_url():
    return f"http://{lan_ip()}:{cfg.PORT}/guard"


def qr_available():
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        return False


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/guard")
def guard():
    return app.send_static_file("guard.html")


@app.route("/guard_qr.png")
def guard_qr():
    if not qr_available():
        return "Install the qrcode package to show a QR code", 404
    import qrcode

    buf = io.BytesIO()
    qrcode.make(guard_url(), box_size=8, border=2).save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/video_feed/<name>")
def video_feed(name):
    worker = workers.get(name)
    if worker is None:
        return "Camera not found", 404

    def stream():
        last = 0
        while worker.running:
            jpeg, seq = worker.wait_frame(last)
            if jpeg is None or seq == last:
                continue
            last = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    with lock:
        return jsonify([w.status() for w in workers.values()])


@app.route("/api/cameras", methods=["POST"])
def create_camera():
    data = request.get_json(silent=True) or {}
    value = str(data.get("value", "")).strip()
    if not value:
        return jsonify({"error": "Enter a camera source."}), 400
    kind = data.get("type", "ip")
    try:
        source = int(value) if kind == "usb" else value
        worker = add_camera(data.get("name"), source, data.get("audio") or None)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(worker.status()), 201


@app.route("/api/cameras/<name>", methods=["DELETE"])
def delete_camera(name):
    with lock:
        worker = workers.pop(name, None)
        if worker is None:
            return jsonify({"error": "Camera not found"}), 404
        worker.stop()
        persist()
    return jsonify({"removed": name})


@app.route("/api/usb_scan")
def usb_scan():
    with lock:
        in_use = sorted(w.source for w in workers.values() if isinstance(w.source, int))
    return jsonify({"available": scan_usb(skip=set(in_use)), "in_use": in_use})


@app.route("/api/alerts")
def get_alerts():
    return jsonify(dispatcher.list())


@app.route("/api/alerts/<int:alert_id>/ack", methods=["POST"])
def ack_alert(alert_id):
    return jsonify({"acked": dispatcher.ack(alert_id)})


@app.route("/api/alerts/<int:alert_id>/action", methods=["POST"])
def alert_action(alert_id):
    data = request.get_json(silent=True) or {}
    alert = dispatcher.act(alert_id, data.get("action", ""), data.get("name"))
    if alert is None:
        return jsonify({"error": "Unknown alert or action"}), 404
    return jsonify(alert)


@app.route("/api/stats")
def stats():
    return jsonify(dispatcher.stats())


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        try:
            cfg.update_settings(data.get("sensitivity"), data.get("verify_first"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    return jsonify(dict(cfg.SETTINGS, verify_timeout=cfg.VERIFY_TIMEOUT))


@app.route("/api/system")
def system():
    with lock:
        demo = any(w.is_demo for w in workers.values())
    return jsonify({"channels": dispatcher.channels(), "thresholds": {"warning": cfg.WARNING_THRESHOLD, "threat": cfg.THREAT_THRESHOLD},
                    "settings": dict(cfg.SETTINGS, verify_timeout=cfg.VERIFY_TIMEOUT), "guard_url": guard_url(),
                    "qr": qr_available(), "demo": demo, "pin": bool(cfg.ACCESS_PIN), "demo_notify": cfg.DEMO_NOTIFY})


@app.route("/api/demo/scenarios")
def demo_scenarios():
    return jsonify([{"id": k, "label": v["label"], "seconds": v["duration"]} for k, v in SCENARIOS.items()])


@app.route("/api/demo/start", methods=["POST"])
def demo_start():
    with lock:
        if not any(w.is_demo for w in workers.values()):
            for w in create_demo_workers(dispatcher):
                workers[w.name] = w
                w.start()
        return jsonify([w.status() for w in workers.values() if w.is_demo])


@app.route("/api/demo/stop", methods=["POST"])
def demo_stop():
    with lock:
        for name in [n for n, w in workers.items() if w.is_demo]:
            workers.pop(name).stop()
    return jsonify({"demo": False})


@app.route("/api/demo/trigger", methods=["POST"])
def demo_trigger():
    data = request.get_json(silent=True) or {}
    with lock:
        demos = [w for w in workers.values() if w.is_demo and hasattr(w, "scene")]
        target = next((w for w in demos if w.name == data.get("camera")), demos[0] if demos else None)
    if target is None:
        return jsonify({"error": "Turn demo mode on first."}), 400
    if data.get("scenario") == "crash" and not data.get("camera"):
        target = next((w for w in demos if w.name == "demo_parking"), target)
    try:
        target.scene.trigger(data.get("scenario", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"camera": target.name, "scenario": data["scenario"]})


@app.route("/api/test_alert", methods=["POST"])
def test_alert():
    """Fires a fake CRITICAL alert so you can check siren, Telegram, Discord, ntfy and MQTT."""
    with lock:
        worker = next((w for w in workers.values() if w.latest_view is not None), None)
    frame = worker.latest_view.copy() if worker else np.zeros((360, 640, 3), dtype=np.uint8)
    cam = worker.name if worker else "test_camera"
    snapshot = alerts_mod.save_snapshot(cam, "TEST_ALERT", frame)
    alert = dispatcher.publish({
        "type": "TEST_ALERT", "service": "POLICE", "severity": "CRITICAL", "camera": cam, "score": 0.99,
        "message": f"Test alert - checking notification channels ({cam})", "reasons": ["manual test from dashboard"],
        "snapshot": snapshot,
    })
    return jsonify(alert)


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(cfg.EVIDENCE_DIR, filename)


def main():
    parser = argparse.ArgumentParser(description="VigiLight")
    parser.add_argument("--camera", action="append", default=[], help="USB index, IP/RTSP URL or video file (repeatable)")
    args = parser.parse_args()
    for c in cfg.load_camera_configs():
        try:
            add_camera(c["name"], c["source"], c.get("audio"), save=False)
        except ValueError as exc:
            logging.warning("skipping saved camera %s: %s", c.get("name"), exc)
    for src in args.camera:
        try:
            add_camera(None, src)
        except ValueError as exc:
            logging.warning("skipping %s: %s", src, exc)
    app.run(host=cfg.HOST, port=cfg.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
