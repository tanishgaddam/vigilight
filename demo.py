"""Demo mode: simulated cameras for presentations, testing and trying VigiLight without hardware.

The scenes are drawn with OpenCV (stick figures, cars) and fed into the REAL behaviour engine,
alert dispatcher, evidence recorder and dashboard. Only the YOLO step is replaced by scripted
detections, so no camera, GPU or model download is needed. Everything is clearly labelled SIMULATED.
"""
import math
import time
import os
import cv2
import numpy as np

from worker import CameraWorker

W, H = 640, 360
FPS = 15.0

SCENARIOS = {
    "attack": {"label": "Armed attack", "duration": 12},
    "fight": {"label": "Street fight", "duration": 12},
    "crash": {"label": "Car crash", "duration": 8},
    "glass": {"label": "Glass breaking", "duration": 4},
    "scream": {"label": "Screaming", "duration": 5},
    "crowd": {"label": "Crowd gathering", "duration": 16},
}


# --- drawing helpers -----------------------------------------------------------
def make_person(pid, cx, cy, h, lw, rw, phase=0.0):
    """Body keypoints (COCO order) + box. lw / rw are wrist offsets from the body centre in body heights."""
    k = np.zeros((17, 3), np.float32)

    def put(i, x, y):
        k[i] = (x, y, 0.9)

    sw = 0.07 * h * math.sin(phase)
    put(0, cx, cy - 0.42 * h)
    put(5, cx - 0.09 * h, cy - 0.25 * h)
    put(6, cx + 0.09 * h, cy - 0.25 * h)
    put(9, cx + lw[0] * h, cy + lw[1] * h)
    put(10, cx + rw[0] * h, cy + rw[1] * h)
    put(7, (k[5][0] + k[9][0]) / 2, (k[5][1] + k[9][1]) / 2 + 0.03 * h)
    put(8, (k[6][0] + k[10][0]) / 2, (k[6][1] + k[10][1]) / 2 + 0.03 * h)
    put(11, cx - 0.06 * h, cy + 0.05 * h)
    put(12, cx + 0.06 * h, cy + 0.05 * h)
    put(13, cx - 0.06 * h + sw * 0.5, cy + 0.28 * h)
    put(14, cx + 0.06 * h - sw * 0.5, cy + 0.28 * h)
    put(15, cx - 0.06 * h + sw, cy + 0.5 * h)
    put(16, cx + 0.06 * h - sw, cy + 0.5 * h)
    return {"id": pid, "box": (cx - 0.22 * h, cy - 0.5 * h, cx + 0.22 * h, cy + 0.5 * h), "kpts": k, "h": h}


def walker(pid, x, y, h, t, offset=0.0):
    a = 0.12 * math.sin(t * 6.0 + offset)
    return make_person(pid, x, y, h, (-0.1, 0.10 + a), (0.1, 0.10 - a), phase=t * 6.0 + offset)


def idle(pid, x, y, h, t=0.0):
    s = 0.01 * math.sin(t * 1.5 + pid)
    return make_person(pid, x, y, h, (-0.09, 0.18 + s), (0.09, 0.18 - s))


def car(cid, cx, cy, w=110, h=55):
    return {"id": cid, "label": "car", "box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)}


def _gradient(top, bottom):
    grad = np.linspace(top, bottom, H).astype(np.uint8)[:, None, :]
    return np.repeat(grad, W, axis=1).copy()


def _glow(img, x, y, radius=110):
    overlay = img.copy()
    cv2.circle(overlay, (x, y), radius, (110, 200, 255), -1)
    return cv2.addWeighted(overlay, 0.16, img, 0.84, 0)


def bg_street():
    img = _gradient((34, 22, 16), (48, 36, 28))
    rng = np.random.default_rng(7)
    x = 0
    while x < W:
        w, h = int(rng.integers(60, 110)), int(rng.integers(90, 170))
        cv2.rectangle(img, (x, 230 - h), (x + w - 4, 230), (44, 34, 30), -1)
        for wy in range(230 - h + 12, 220, 22):
            for wx in range(x + 8, x + w - 14, 20):
                if rng.random() < 0.45:
                    cv2.rectangle(img, (wx, wy), (wx + 9, wy + 11), (70, 170, 225), -1)
        x += w
    cv2.rectangle(img, (0, 230), (W, 290), (74, 70, 66), -1)
    cv2.rectangle(img, (0, 286), (W, 292), (110, 106, 100), -1)
    cv2.rectangle(img, (0, 292), (W, H), (42, 42, 44), -1)
    for lx in range(0, W, 70):
        cv2.line(img, (lx, 332), (lx + 34, 332), (150, 150, 150), 2)
    cv2.line(img, (560, 290), (560, 120), (95, 95, 95), 4)
    cv2.circle(img, (560, 118), 7, (170, 230, 255), -1)
    return _glow(img, 560, 118)


def bg_parking():
    img = _gradient((30, 28, 28), (50, 50, 52))
    cv2.rectangle(img, (0, 0), (W, 110), (36, 34, 36), -1)
    for fx in range(0, W, 14):
        cv2.line(img, (fx, 20), (fx, 110), (60, 60, 64), 1)
    for bx in range(20, W, 120):
        cv2.line(img, (bx, 150), (bx, 232), (170, 170, 170), 2)
    cv2.line(img, (0, 262), (W, 262), (120, 120, 120), 1)
    for lx in range(0, W, 70):
        cv2.line(img, (lx, 300), (lx + 34, 300), (140, 140, 140), 2)
    cv2.line(img, (120, 110), (120, 55), (95, 95, 95), 4)
    cv2.line(img, (520, 110), (520, 55), (95, 95, 95), 4)
    img = _glow(img, 120, 52, 120)
    return _glow(img, 520, 52, 120)


def draw_person(img, p):
    k = p["kpts"]
    h = p["h"]
    c = (232, 232, 236)
    t = max(2, int(h / 60))

    def line(a, b):
        cv2.line(img, tuple(int(v) for v in k[a][:2]), tuple(int(v) for v in k[b][:2]), c, t, cv2.LINE_AA)

    mid_sh = ((k[5][0] + k[6][0]) / 2, (k[5][1] + k[6][1]) / 2)
    mid_hip = ((k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2)
    cv2.line(img, (int(mid_sh[0]), int(mid_sh[1])), (int(mid_hip[0]), int(mid_hip[1])), c, t + 1, cv2.LINE_AA)
    for a, b in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
        line(a, b)
    cv2.circle(img, (int(k[0][0]), int(k[0][1])), max(5, int(h * 0.07)), c, -1, cv2.LINE_AA)


def draw_knife(img, wx, wy, h):
    cv2.line(img, (int(wx), int(wy)), (int(wx + 0.03 * h), int(wy - 0.17 * h)), (210, 220, 230), 3, cv2.LINE_AA)
    cv2.line(img, (int(wx), int(wy)), (int(wx - 0.01 * h), int(wy + 0.04 * h)), (40, 60, 120), 4, cv2.LINE_AA)


def draw_car(img, v, color):
    x1, y1, x2, y2 = (int(n) for n in v["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cw = (x2 - x1)
    cv2.rectangle(img, (x1 + int(cw * 0.22), y1 + 6), (x1 + int(cw * 0.78), y1 + int((y2 - y1) * 0.5)), (190, 210, 225), -1)
    for wx in (x1 + 14, x2 - 14):
        cv2.circle(img, (wx, y2), 8, (20, 20, 20), -1)


class DemoScene:
    def __init__(self, bg):
        self.bg_kind = bg
        self.bg = bg_street() if bg == "street" else bg_parking()
        self.t0 = time.time()
        self.active = None
        self.audio_cb = None

    # --- control ---------------------------------------------------------------
    def trigger(self, name):
        if name not in SCENARIOS:
            raise ValueError("unknown scenario")
        self.active = {"name": name, "t0": time.time(), "fired": set()}

    def info(self):
        act = self.active
        if not act:
            return None
        left = SCENARIOS[act["name"]]["duration"] - (time.time() - act["t0"])
        return {"name": act["name"], "label": SCENARIOS[act["name"]]["label"], "remaining": max(0, round(left))}

    def _audio(self, act, key, at, scores, e):
        if e >= at and key not in act["fired"]:
            act["fired"].add(key)
            if self.audio_cb:
                self.audio_cb(scores)

    # --- scene content -----------------------------------------------------------
    def _ambient(self, t):
        if self.bg_kind == "street":
            span = W + 160
            return [walker(1, (100 + 45 * t) % span - 80, 262, 140, t),
                    walker(2, (380 - 55 * t) % span - 80, 272, 148, t, 1.0),
                    walker(3, (560 + 38 * t) % span - 80, 256, 132, t, 2.0)]
        return [walker(5, (200 + 40 * t) % (W + 160) - 80, 246, 135, t)]

    def _parked(self):
        return [car(61, 80, 190), car(62, 320, 190), car(63, 560, 190)] if self.bg_kind == "parking" else []

    def _build(self, now):
        act = self.active
        if act and now - act["t0"] > SCENARIOS[act["name"]]["duration"]:
            self.active = act = None
        t = now - self.t0
        e = now - act["t0"] if act else 0.0
        name = act["name"] if act else None
        persons, weapons, vehicles = [], [], self._parked()
        caption = "calm scene"
        fx = None
        lane_y = 300 if self.bg_kind == "parking" else 318

        if name in (None, "glass", "scream"):
            persons = self._ambient(t)
        if name == "glass":
            caption = "glass breaking (audio only)"
            self._audio(act, "g", 0.8, {"GLASS_BREAK": 0.84}, e)
            fx = ("glass", e)
        elif name == "scream":
            caption = "screaming (audio only)"
            self._audio(act, "s1", 0.8, {"SCREAM": 0.86}, e)
            self._audio(act, "s2", 1.9, {"SCREAM": 0.88}, e)
            fx = ("sound", e)
        elif name == "attack":
            caption = "armed attack"
            vx, vy, vh = 430, 252, 145
            persons.append(idle(12, vx, vy, vh, t))
            ax = 70 + 40 * e if e < 2.0 else min(300.0, 150 + 190 * (e - 2.0))
            swing = 0.35 * math.sin(2 * math.pi * 2.5 * e)
            rw = (0.2 + 0.1 * swing, -0.30 + swing * 0.3) if e >= 1.5 else (0.12, 0.12)
            a = make_person(11, ax, vy, vh, (-0.12, 0.12), rw, phase=e * 6.0)
            persons.append(a)
            wx, wy = a["kpts"][10][:2]
            weapons.append({"label": "knife", "box": (float(wx - 8), float(wy - 22), float(wx + 8), float(wy + 22)), "conf": 0.7, "_draw": (wx, wy, vh)})
            self._audio(act, "scream", 3.8, {"SCREAM": 0.82}, e)
        elif name == "fight":
            caption = "street fight"
            sway = 8 * math.sin(e * 5) if e >= 1.5 else 0.0
            fighting = e >= 1.5
            s1 = 0.4 * math.sin(2 * math.pi * 2.5 * e) if fighting else 0.0
            s2 = 0.4 * math.sin(2 * math.pi * 2.5 * e + 1.3) if fighting else 0.0
            persons.append(make_person(21, 270 + sway, 252, 145, (-0.1, s1), (0.2 + s1 * 0.5, -0.1 + s1), phase=e * 6))
            persons.append(make_person(22, 335 - sway, 252, 145, (-0.2 - s2 * 0.5, -0.1 + s2), (0.1, s2), phase=e * 6 + 1))
            persons.append(idle(23, 90, 262, 138, t))
            persons.append(idle(24, 560, 258, 134, t))
        elif name == "crash":
            caption = "car crash"
            tt = min(e, 1.0)
            vehicles += [car(31, 60 + 230 * tt, lane_y), car(32, 580 - 230 * tt, lane_y)]
            fx = ("impact", e, (60 + 230 * tt + 580 - 230 * tt) / 2, lane_y)
        elif name == "crowd":
            caption = "crowd gathering"
            for i in range(12):
                x = 60 + (i % 6) * 100 + 18 * math.sin(0.7 * t + i)
                y = 232 + (i // 6) * 52 + 6 * math.cos(0.5 * t + i)
                persons.append(walker(40 + i, x, y, 118 + 12 * (i // 6) + 4 * (i % 3), t * 0.3, i))
        return {"persons": persons, "weapons": weapons, "vehicles": vehicles, "caption": caption, "fx": fx}

    # --- rendering -----------------------------------------------------------------
    def render(self, now):
        data = self._build(now)
        img = self.bg.copy()
        for i, v in enumerate(data["vehicles"]):
            draw_car(img, v, ((90, 90, 160), (140, 110, 80), (80, 130, 120), (70, 70, 70))[i % 4])
        for p in sorted(data["persons"], key=lambda q: q["box"][3]):
            draw_person(img, p)
        for w in data["weapons"]:
            wx, wy, h = w["_draw"]
            draw_knife(img, wx, wy, h)
        fx = data["fx"]
        if fx and fx[0] == "glass" and fx[1] >= 0.8:
            age = fx[1] - 0.8
            if age < 1.6:
                for ang in range(0, 360, 40):
                    r = 20 + 60 * age
                    cv2.line(img, (430, 150), (int(430 + r * math.cos(math.radians(ang))), int(150 + r * math.sin(math.radians(ang)))), (235, 245, 255), 1, cv2.LINE_AA)
        if fx and fx[0] == "sound":
            for r in (24, 44, 64):
                cv2.circle(img, (320, 110), int(r + 6 * math.sin(fx[1] * 8)), (80, 80, 240), 2, cv2.LINE_AA)
        if fx and fx[0] == "impact" and 1.0 <= fx[1] < 1.6:
            cv2.circle(img, (int(fx[2]), int(fx[3])), int(26 + 30 * (fx[1] - 1.0)), (40, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "SIMULATED: " + data["caption"], (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (W - 78, 8), (W - 8, 30), (30, 30, 30), -1)
        cv2.putText(img, "DEMO", (W - 66, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
        return img, data


class SyntheticVideo:
    """Quacks like VideoSource, but renders the scene instead of reading a camera."""

    def __init__(self, scene):
        self.scene = scene
        self.online = True
        self.fps = FPS
        self.current = {"persons": [], "weapons": [], "vehicles": []}
        self._idx = -1
        self._frame = None
        self._t0 = time.time()

    def start(self):
        pass

    def stop(self):
        pass

    def get(self):
        now = time.time()
        idx = int((now - self._t0) * FPS)
        if idx != self._idx:
            self._idx = idx
            self._frame, self.current = self.scene.render(now)
        return self._frame, idx + 1


class SyntheticDetector:
    def __init__(self, video):
        self.video = video

    def detect(self, frame):
        c = self.video.current
        return {"persons": list(c["persons"]), "weapons": list(c["weapons"]), "vehicles": list(c["vehicles"])}


DEMO_CAMERAS = (("demo_main_street", "street"), ("demo_parking", "parking"))
DEMO_VIDEO = r"C:\Users\ASUS\Downloads\6296295-hd_1080_1920_25fps.mp4"


def create_demo_workers(dispatcher):
    workers = []
    for name, bg in DEMO_CAMERAS:
        scene = DemoScene(bg)
        video = SyntheticVideo(scene)
        w = CameraWorker(name, "demo:" + bg, "off", dispatcher, video=video, detector=SyntheticDetector(video), is_demo=True)
        scene.audio_cb = w.on_audio_scores
        w.scene = scene
        workers.append(w)
    if os.path.isfile(DEMO_VIDEO):
        workers.append(CameraWorker("demo_fight_clip", DEMO_VIDEO, "off", dispatcher, is_demo=True))
    return workers