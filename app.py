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

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_from_directory

import alerts as alerts_mod
import config as cfg
from alerts import AlertDispatcher
from demo import create_demo_worker, list_demo_videos
from sources import scan_usb
from worker import CameraWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", static_url_path="/static")
dispatcher = AlertDispatcher()
workers = {}
lock = threading.RLock()
demo_mode = False   # True after "Start Demo"; nothing plays until a video is picked in the dropdown


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
    return hasattr(cv2, "QRCodeEncoder_create")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/guard")
def guard():
    return app.send_static_file("guard.html")


@app.route("/guard_qr.png")
def guard_qr():
    try:
        enc = cv2.QRCodeEncoder_create()
        qr = enc.encode(guard_url())
        img = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", img)
        if ok:
            return Response(buf.tobytes(), mimetype="image/png")
    except Exception as exc:
        logging.warning("qr code generation failed: %s", exc)
    return "QR generation failed", 500


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


@app.route("/api/cameras/<name>/pause", methods=["POST"])
def toggle_camera_pause(name):
    with lock:
        worker = workers.get(name)
        if worker is None:
            return jsonify({"error": "Camera not found"}), 404
        paused = worker.toggle_pause()
    return jsonify({"name": name, "paused": paused})


@app.route("/api/cameras/<name>/xray", methods=["POST"])
def toggle_camera_xray(name):
    with lock:
        worker = workers.get(name)
        if worker is None:
            return jsonify({"error": "Camera not found"}), 404
        xray = worker.toggle_xray()
    return jsonify({"name": name, "xray": xray})


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    videos_dir = cfg.BASE_DIR / "videos"
    videos_dir.mkdir(exist_ok=True)
    filename = cfg.safe_name(file.filename.rsplit(".", 1)[0]) + "." + (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "mp4")
    dest = videos_dir / filename
    file.save(str(dest))
    cam_name = request.form.get("name") or filename.rsplit(".", 1)[0]
    try:
        worker = add_camera(cam_name, str(dest), audio="off")
        return jsonify(worker.status()), 201
    except ValueError as exc:
        return jsonify({"error": str(exc), "path": str(dest)}), 400


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
    data = request.get_json(silent=True) or {}
    name = data.get("name") or "Command Center"
    return jsonify({"acked": dispatcher.ack(alert_id, name=name)})


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
        demo = demo_mode or any(w.is_demo for w in workers.values())
    return jsonify({"channels": dispatcher.channels(), "thresholds": {"warning": cfg.WARNING_THRESHOLD, "threat": cfg.THREAT_THRESHOLD},
                    "settings": dict(cfg.SETTINGS, verify_timeout=cfg.VERIFY_TIMEOUT), "guard_url": guard_url(),
                    "qr": qr_available(), "demo": demo, "pin": bool(cfg.ACCESS_PIN), "demo_notify": cfg.DEMO_NOTIFY})


@app.route("/api/demo/scenarios")
def demo_scenarios():
    """Videos available as demo cameras (same list the dropdown uses)."""
    return jsonify([{"id": v["id"], "label": v["label"], "camera": v["id"]} for v in list_demo_videos()])


@app.route("/api/demo/videos")
def demo_videos():
    """Video files found in videos/ (and Downloads) for the dropdown; 'playing' marks the active one."""
    with lock:
        playing = {n for n, w in workers.items() if w.is_demo}
    return jsonify([dict(v, playing=v["id"] in playing) for v in list_demo_videos()])


@app.route("/api/demo/start", methods=["POST"])
def demo_start():
    """Turn demo mode on. With {"video": "<id>"} only that video plays (other demo videos are stopped).
    Without a video nothing is started - pick one from the dropdown."""
    global demo_mode
    data = request.get_json(silent=True) or {}
    wanted = data.get("video") or data.get("id") or data.get("camera")
    with lock:
        demo_mode = True
        if wanted:
            video = next((v for v in list_demo_videos() if v["id"] == wanted), None)
            if video is None:
                return jsonify({"error": "Video not found. Put it in the videos/ folder."}), 404
            for name in [n for n, w in workers.items() if w.is_demo and n != wanted]:
                workers.pop(name).stop()
            if wanted in workers:
                workers[wanted].video.restart()        # already loaded: replay from the start
            else:
                w = create_demo_worker(dispatcher, video)
                workers[w.name] = w
                w.start()
        return jsonify([w.status() for w in workers.values() if w.is_demo])


@app.route("/api/demo/stop", methods=["POST"])
def demo_stop():
    global demo_mode
    with lock:
        demo_mode = False
        for name in [n for n, w in workers.items() if w.is_demo]:
            workers.pop(name).stop()
    return jsonify({"demo": False})


@app.route("/api/demo/trigger", methods=["POST"])
def demo_trigger():
    """Play (unpause/restart) a specific demo camera by name, and pause others to protect CPU."""
    data = request.get_json(silent=True) or {}
    camera_name = data.get("camera") or data.get("id")
    with lock:
        demo_workers = [w for w in workers.values() if w.is_demo]
        if not demo_workers:
            return jsonify({"error": "Turn demo mode on first."}), 400
        # Find the specific camera, or use first if not specified
        target = next((w for w in demo_workers if w.name == camera_name), demo_workers[0])
        # Pause all OTHER demo workers so CPU can focus full AI inference on this video
        for w in demo_workers:
            if w.name != target.name and hasattr(w, "pause"):
                w.pause()
    # Restart target video from beginning (unpause + seek to frame 0)
    if hasattr(target, "video") and hasattr(target.video, "restart"):
        target.video.restart()
    elif hasattr(target, "resume"):
        target.resume()
    return jsonify({"camera": target.name, "playing": True})


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


@app.route("/api/test_audio", methods=["POST"])
def test_audio():
    """Trigger an audio threat detection test (glass break, scream, or gunshot)."""
    data = request.get_json(silent=True) or {}
    sound_type = str(data.get("type", "glass")).lower()
    with lock:
        worker = next((w for w in workers.values() if w.latest_view is not None), None)
    cam = worker.name if worker else "main_mic"
    frame = worker.latest_view.copy() if worker else np.zeros((360, 640, 3), dtype=np.uint8)
    if "scream" in sound_type:
        alert = dispatcher.publish({
            "type": "DISTRESS_SCREAM", "service": "POLICE", "severity": "HIGH", "camera": cam, "score": 0.91,
            "message": f"Screaming or distress shouting detected on microphone ({cam})",
            "reasons": ["Acoustic signature: vocal distress band (1.4 kHz)", "Sustained vocal resonance: 0.91"],
            "snapshot": alerts_mod.save_snapshot(cam, "DISTRESS_SCREAM", frame),
        })
    elif "gun" in sound_type:
        alert = dispatcher.publish({
            "type": "GUNSHOT_HEARD", "service": "POLICE", "severity": "CRITICAL", "camera": cam, "score": 0.98,
            "message": f"Gunshot or explosive blast detected on microphone ({cam})",
            "reasons": ["Acoustic signature: explosive transient impulse", "Crest factor: 8.4"],
            "snapshot": alerts_mod.save_snapshot(cam, "GUNSHOT", frame),
        })
    else:
        alert = dispatcher.publish({
            "type": "GLASS_BREAK_HEARD", "service": "POLICE", "severity": "HIGH", "camera": cam, "score": 0.95,
            "message": f"Glass breaking sound detected on microphone ({cam})",
            "reasons": ["Acoustic signature: high-frequency shatter (4.2 kHz)", "Transient burst ratio: 0.95"],
            "snapshot": alerts_mod.save_snapshot(cam, "GLASS_BREAK", frame),
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
