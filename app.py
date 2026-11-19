"""VigiLight - AI street-light surveillance. Run:  python app.py  then open http://localhost:5000

Cameras can be added in the dashboard, or on the command line:
    python app.py --camera 0 --camera http://192.168.1.20:8080/video --camera rtsp://user:pw@host/stream
"""
import argparse
import logging
import threading

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

import alerts as alerts_mod
import config as cfg
from alerts import AlertDispatcher
from sources import scan_usb
from worker import CameraWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", static_url_path="/static")
dispatcher = AlertDispatcher()
workers = {}
lock = threading.RLock()


def persist():
    cfg.save_camera_configs([{"name": w.name, "source": w.source, "audio": w.audio_cfg} for w in workers.values()])


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


@app.route("/")
def index():
    return app.send_static_file("index.html")


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


@app.route("/api/system")
def system():
    return jsonify({"channels": dispatcher.channels(), "thresholds": {"warning": cfg.WARNING_THRESHOLD, "threat": cfg.THREAT_THRESHOLD}})


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
