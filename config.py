"""VigiLight configuration.

Secrets and tunables come from environment variables (or a .env file next to this
file). Nothing sensitive is hardcoded.
"""
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # python-dotenv is optional
    pass


def _str(name, default=""):
    return os.getenv(name, default).strip()


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- server ---------------------------------------------------------------
HOST = _str("VIGILIGHT_HOST", "0.0.0.0")
PORT = _int("VIGILIGHT_PORT", 5000)

# --- models ---------------------------------------------------------------
OBJECT_MODEL = _str("OBJECT_MODEL", "yolo11n.pt")        # vehicles + COCO weapons
POSE_MODEL = _str("POSE_MODEL", "yolo11n-pose.pt")       # people + body keypoints
WEAPON_MODEL = _str("WEAPON_MODEL", "")                  # optional fine-tuned weapon model
DEVICE = _str("DEVICE", "")
IMG_SIZE = _int("IMG_SIZE", 640)
PROCESS_FPS = _float("PROCESS_FPS", 8)
PERSON_CONF = _float("PERSON_CONF", 0.40)
OBJECT_CONF = _float("OBJECT_CONF", 0.30)
WEAPON_CONF = _float("WEAPON_CONF", 0.45)
STREAM_WIDTH = _int("STREAM_WIDTH", 640)

# --- behaviour scoring ----------------------------------------------------
WARNING_THRESHOLD = _float("WARNING_THRESHOLD", 0.35)    # dashboard shows WARNING
THREAT_THRESHOLD = _float("THREAT_THRESHOLD", 0.60)      # alert is raised (after sustain time)
SUSTAIN_ARMED = _float("SUSTAIN_ARMED", 0.8)             # seconds above threshold, armed person
SUSTAIN_UNARMED = _float("SUSTAIN_UNARMED", 1.5)         # seconds above threshold, fight
SMOOTHING = _float("SMOOTHING", 0.5)                     # EMA weight of the newest score
STRIKE_MIN_SPEED = _float("STRIKE_MIN_SPEED", 0.8)       # body-heights/sec of arm motion = normal movement
STRIKE_SPEED = _float("STRIKE_SPEED", 2.5)               # body-heights/sec of arm motion = full strike
APPROACH_SPEED = _float("APPROACH_SPEED", 1.5)           # body-heights/sec closing speed = full approach
PROX_NEAR = _float("PROX_NEAR", 0.8)                     # distance (body heights) that counts as "right next to"
PROX_FAR = _float("PROX_FAR", 3.0)                       # distance beyond which proximity is ignored
ALERT_COOLDOWN = _float("ALERT_COOLDOWN", 15)            # seconds between alerts of the same type per camera

# --- audio ----------------------------------------------------------------
AUDIO_ENABLED = _bool("AUDIO_ENABLED", True)
AUDIO_THRESHOLD = _float("AUDIO_THRESHOLD", 0.5)
AUDIO_CONTEXT_SEC = _float("AUDIO_CONTEXT_SEC", 4.0)     # how long a sound boosts nearby visual threat scores
SCREAM_HITS = _int("SCREAM_HITS", 2)                     # consecutive audio windows needed for a scream alert

# --- evidence -------------------------------------------------------------
EVIDENCE_DIR = Path(_str("EVIDENCE_DIR", str(BASE_DIR / "evidence")))
SAVE_CLIPS = _bool("SAVE_CLIPS", True)
CLIP_PRE_SEC = _float("CLIP_PRE_SEC", 4)
CLIP_POST_SEC = _float("CLIP_POST_SEC", 3)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# --- alert channels -------------------------------------------------------
TELEGRAM_BOT_TOKEN = _str("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _str("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = _str("DISCORD_WEBHOOK_URL")
NTFY_TOPIC = _str("NTFY_TOPIC")
NTFY_SERVER = _str("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
MQTT_ENABLED = _bool("MQTT_ENABLED", True)
MQTT_BROKER = _str("MQTT_BROKER", "broker.hivemq.com")
MQTT_TOPIC = _str("MQTT_TOPIC", "sfit/vigilight/alerts")

# --- camera persistence ---------------------------------------------------
CAMERAS_FILE = BASE_DIR / "cameras.json"
_file_lock = threading.Lock()


def load_camera_configs():
    if not CAMERAS_FILE.exists():
        return []
    try:
        data = json.loads(CAMERAS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [c for c in data if isinstance(c, dict) and "name" in c and "source" in c]


def save_camera_configs(configs):
    with _file_lock:
        CAMERAS_FILE.write_text(json.dumps(configs, indent=2), encoding="utf-8")


def safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())[:32].strip("_")
    return cleaned or "camera"


def normalize_source(raw):
    """Turn whatever the user typed into something OpenCV can open.

    0 / "usb:1"                      -> USB webcam index
    "192.168.1.5:8080"               -> http://192.168.1.5:8080/video   (IP Webcam style apps)
    "rtsp://user:pw@host/stream"     -> unchanged
    "C:/clips/test.mp4"              -> video file (loops, handy for demos and evaluation)
    """
    if isinstance(raw, int):
        return raw
    raw = str(raw).strip()
    if raw.lower().startswith("usb:"):
        raw = raw[4:].strip()
    if raw.isdigit():
        return int(raw)
    if os.path.isfile(raw):
        return raw
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https") and parsed.path in ("", "/"):
        raw = raw.rstrip("/") + "/video"
    return raw


def derive_audio_source(source, audio):
    """audio: off | auto | mic | mic:<device index> | <explicit http url to a WAV stream>"""
    if audio in (None, "", "off", False):
        return None
    if audio == "auto":
        if isinstance(source, str) and source.startswith("http") and source.endswith("/video"):
            return source[: -len("/video")] + "/audio.wav"
        return None
    return str(audio)
