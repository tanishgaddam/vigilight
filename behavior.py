"""Behaviour analysis: turns raw detections into threat scores and alerts.

A weapon on its own never raises an alert. For each tracked person we combine several
signals over time and only alert when the combined score stays high:

  armed person:   weapon in hand + raised arm + fast arm motion + close to / closing in
                  on another person
  unarmed fight:  fast arm motion near another person, the other person moving
                  aggressively too, closing distance
  audio:          screams / breaking glass / gunshots nearby boost the visual score

This module has no ML imports, so it can be unit-tested with synthetic data.
Distances and speeds are measured in "body heights" so they work at any camera distance.
"""
import math
from collections import deque
from itertools import combinations

import numpy as np

import config as cfg

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10  # COCO keypoint indexes
KP_CONF = 0.35
JITTER = 0.015          # ignore keypoint wobble smaller than this (body heights)
STRIKE_WINDOW = 1.0     # seconds (expanded to handle low FPS / multi-camera CPU load)
APPROACH_WINDOW = 1.5   # seconds (expanded to handle low FPS / multi-camera CPU load)
ARMED_HOLD_SEC = 1.2    # keep "armed" status through brief weapon-detection dropouts
LEVEL_RANK = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "THREAT": 3}
AUDIO_NAMES = {
    "SCREAM": "screaming/shouting",
    "GLASS_BREAK": "glass breaking",
    "GUNSHOT": "gunshot-like sound",
}


def clamp01(x):
    return max(0.0, min(1.0, x))


def _wrists(kpts):
    out = np.full((2, 2), np.nan, dtype=np.float32)
    if kpts is None:
        return out
    for k, idx in enumerate((L_WRIST, R_WRIST)):
        if kpts[idx][2] > KP_CONF:
            out[k] = kpts[idx][:2]
    return out


def _raised(kpts):
    """True if a wrist is above shoulder height."""
    if kpts is None:
        return False
    sh = [kpts[i] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] > KP_CONF]
    if not sh:
        return False
    shoulder_y = sum(s[1] for s in sh) / len(sh)
    return any(kpts[w][2] > KP_CONF and kpts[w][1] < shoulder_y for w in (L_WRIST, R_WRIST))


def _inside(x, y, box, margin=0.0):
    m = margin * max(box[2] - box[0], box[3] - box[1])
    return box[0] - m <= x <= box[2] + m and box[1] - m <= y <= box[3] + m


def _holding(person, weapons):
    """(0..1 confidence that this person holds a weapon, the weapon)."""
    if not weapons:
        return 0.0, None
    box = person["box"]
    height = max(1.0, box[3] - box[1])
    kpts = person["kpts"]
    best, best_w = 0.0, None
    for w in weapons:
        wx, wy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
        if kpts is not None:
            for idx in (L_WRIST, R_WRIST):
                x, y, c = kpts[idx]
                if c > KP_CONF and (math.hypot(x - wx, y - wy) / height < 0.30 or _inside(x, y, w["box"], 0.25)):
                    return 1.0, w          # weapon is at the hand
        if _inside(wx, wy, box) and best < 0.5:
            best, best_w = 0.5, w          # weapon is somewhere on the person
    return best, best_w


class PersonState:
    def __init__(self):
        self.hist = deque(maxlen=40)       # (t, wrists relative to body, in body heights)
        self.dist_hist = deque(maxlen=40)  # (t, distance to nearest other person)
        self.score = 0.0
        self.level = "NORMAL"
        self.above_since = None
        self.peak = 0.0
        self.peak_signals = []
        self.peak_reasons = []
        self.cooldown_until = 0.0
        self.last_seen = 0.0
        self.armed_until = 0.0
        self.armed_val = 0.0
        self.weapon_label = "weapon"


def _strike(st, now):
    """0..1 measure of fast arm movement relative to the body (walking is subtracted)."""
    pts = [(t, rel) for t, rel in st.hist if now - t <= STRIKE_WINDOW]
    if len(pts) < 2:
        return 0.0
    dt = pts[-1][0] - pts[0][0]
    if dt < 0.05:
        return 0.0
    best = 0.0
    for k in range(2):
        path, prev = 0.0, None
        for _, rel in pts:
            q = rel[k]
            if np.isnan(q).any():
                prev = None
                continue
            if prev is not None:
                step = float(np.linalg.norm(q - prev))
                if step > JITTER:
                    path += step
            prev = q
        best = max(best, path / dt)
    span = max(1e-6, cfg.STRIKE_SPEED - cfg.STRIKE_MIN_SPEED)
    return clamp01((best - cfg.STRIKE_MIN_SPEED) / span)


def _approach(st, now):
    pts = [(t, d) for t, d in st.dist_hist if now - t <= APPROACH_WINDOW]
    if len(pts) < 2:
        return 0.0
    dt = pts[-1][0] - pts[0][0]
    if dt < 0.05:
        return 0.0
    return clamp01(((pts[0][1] - pts[-1][1]) / dt) / cfg.APPROACH_SPEED)


class ThreatEngine:
    def __init__(self):
        self.states = {}

    def update(self, persons, weapons, audio, now):
        """persons: [{id, box, kpts}], weapons: [{label, box, conf}],
        audio: None or {label, score}. Returns (per-person results, alerts)."""
        infos = {}
        for p in persons:
            pid = p["id"]
            st = self.states.setdefault(pid, PersonState())
            st.last_seen = now
            x1, y1, x2, y2 = p["box"]
            h = max(1.0, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rel = (_wrists(p["kpts"]) - np.array([cx, cy], dtype=np.float32)) / h
            st.hist.append((now, rel))
            armed, weapon = _holding(p, weapons)
            if armed > 0:
                st.armed_until, st.armed_val, st.weapon_label = now + ARMED_HOLD_SEC, armed, weapon["label"]
            elif now < st.armed_until:
                armed = st.armed_val * 0.8
            infos[pid] = {"p": p, "st": st, "h": h, "c": (cx, cy), "armed": armed,
                          "raised": _raised(p["kpts"]), "strike": _strike(st, now)}

        results, alerts = [], []
        for pid, me in infos.items():
            st = me["st"]
            tid, tgt, dist = None, None, None
            for oid, other in infos.items():
                if oid == pid:
                    continue
                d = math.hypot(me["c"][0] - other["c"][0], me["c"][1] - other["c"][1]) / ((me["h"] + other["h"]) / 2)
                if dist is None or d < dist:
                    tid, tgt, dist = oid, other, d
            if tgt is None:
                st.dist_hist.clear()
                prox = approach = other_strike = 0.0
            else:
                st.dist_hist.append((now, dist))
                prox = clamp01((cfg.PROX_FAR - dist) / max(1e-6, cfg.PROX_FAR - cfg.PROX_NEAR))
                approach = _approach(st, now)
                other_strike = tgt["strike"]

            armed, strike = me["armed"], me["strike"]
            reasons, parts = [], []   # parts = plain-language signal breakdown for "Why did this trigger?"
            if armed > 0:
                kind = "ARMED"
                raw = 0.20 * armed
                parts.append(("Weapon in hand", 0.20 * armed, 0.20))
                reasons.append(f"{st.weapon_label} in hand")
                raised_v = 0.15 * armed if me["raised"] else 0.0
                raw += raised_v
                parts.append(("Arm raised", raised_v, 0.15))
                if me["raised"]:
                    reasons.append("arm raised")
                strike_v = 0.20 * strike if strike > 0.3 else 0.0
                raw += strike_v
                parts.append(("Fast arm movement", strike_v, 0.20))
                if strike > 0.3:
                    reasons.append("fast arm motion")
                if tgt is not None:
                    raw += 0.20 * prox + 0.25 * approach
                    parts.append(("Close to another person", 0.20 * prox, 0.20))
                    parts.append(("Moving toward them", 0.25 * approach, 0.25))
                    if prox > 0.5:
                        reasons.append(f"right next to person #{tid}")
                    if approach > 0.4:
                        reasons.append(f"closing in on person #{tid}")
            else:
                kind = "VIOLENCE"
                raw = 0.0
                if tgt is not None and prox >= 0.15:
                    approach_v = 0.15 * approach * min(1.0, 2 * strike)
                    raw = 0.40 * strike + 0.25 * other_strike + 0.20 * prox + approach_v
                    raw = min(raw, 0.9)
                    parts += [("Fast arm movement", 0.40 * strike, 0.40),
                              ("Other person also aggressive", 0.25 * other_strike, 0.25),
                              ("Close together", 0.20 * prox, 0.20),
                              ("Moving toward them", approach_v, 0.15)]
                    if strike > 0.3:
                        reasons.append(f"fast arm motion next to person #{tid}")
                    if other_strike > 0.3:
                        reasons.append(f"person #{tid} also moving aggressively")
            if audio and raw >= 0.2:
                raw = min(1.0, raw + 0.25 * audio["score"])
                parts.append(("Alarming sound nearby", 0.25 * audio["score"], 0.25))
                reasons.append(f"audio: {AUDIO_NAMES.get(audio['label'], audio['label'])} ({audio['score']:.2f})")
            signals = [{"name": n, "value": round(v, 3), "max": m} for n, v, m in parts]

            st.score = (1 - cfg.SMOOTHING) * st.score + cfg.SMOOTHING * raw
            if st.score >= cfg.THREAT_THRESHOLD:
                st.level = "THREAT"
            elif st.score >= cfg.WARNING_THRESHOLD:
                st.level = "WARNING"
            elif armed > 0:
                st.level = "WATCH"      # weapon visible, nothing aggressive: log only, no alert
            else:
                st.level = "NORMAL"

            # Hysteresis: the timer starts at the threat threshold but only resets once the
            # score falls clearly below it, so a noisy score hovering near 0.6 still counts.
            if st.score >= cfg.THREAT_THRESHOLD and st.above_since is None:
                st.above_since = now
                st.peak, st.peak_signals, st.peak_reasons = 0.0, [], []
            elif st.score < cfg.THREAT_THRESHOLD * 0.85:
                st.above_since = None
            if st.above_since is not None and st.score >= st.peak:   # remember the strongest moment for the alert
                st.peak, st.peak_signals, st.peak_reasons = st.score, signals, reasons
            if st.above_since is not None:
                need = cfg.SUSTAIN_ARMED if kind == "ARMED" else cfg.SUSTAIN_UNARMED
                if now - st.above_since >= need and now >= st.cooldown_until:
                    st.cooldown_until = now + cfg.ALERT_COOLDOWN
                    if tgt is not None and kind == "VIOLENCE" and tgt["st"].level == "THREAT":
                        tgt["st"].cooldown_until = now + cfg.ALERT_COOLDOWN  # one alert per fight, not one per fighter
                    near = tgt is not None and prox > 0.3
                    if kind == "ARMED":
                        msg = f"Armed person #{pid} behaving aggressively" + (f" toward person #{tid}" if near else "")
                        alerts.append({"type": "ARMED_ATTACK_RISK", "service": "POLICE", "severity": "CRITICAL",
                                       "message": msg, "score": round(st.peak, 2), "reasons": st.peak_reasons, "person_id": pid,
                                       "signals": st.peak_signals, "threshold": cfg.THREAT_THRESHOLD, "sustain": need})
                    else:
                        msg = f"Possible physical fight involving person #{pid}" + (f" and #{tid}" if near else "")
                        alerts.append({"type": "VIOLENCE_DETECTED", "service": "POLICE", "severity": "HIGH",
                                       "message": msg, "score": round(st.peak, 2), "reasons": st.peak_reasons, "person_id": pid,
                                       "signals": st.peak_signals, "threshold": cfg.THREAT_THRESHOLD, "sustain": need})

            results.append({"id": pid, "box": me["p"]["box"], "kpts": me["p"]["kpts"], "level": st.level,
                            "score": st.score, "armed": armed > 0, "weapon": st.weapon_label if armed > 0 else None,
                            "signals": signals})

        for pid in [k for k, s in self.states.items() if now - s.last_seen > 3.0]:
            del self.states[pid]
        return results, alerts


def _overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / area if area > 0 else 0.0


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CollisionEngine:
    """Road accident = two tracked vehicles overlap while at least one was moving
    beforehand, and then both stay (almost) stopped. Parked or jammed traffic that
    merely overlaps in the image does not trigger it."""

    MOVING = 0.35        # diagonals/sec before contact
    LOW = 0.12           # diagonals/sec after contact
    CONTACT = 0.12       # overlap ratio that counts as contact (strict mode)
    CONFIRM_SEC = 0.7    # (strict mode)
    DUPLICATE_IOU = 0.80  # two boxes this similar are the same object detected twice, not two vehicles

    def __init__(self):
        self.tracks = {}
        self.pairs = {}

    @staticmethod
    def _speed(pts, now, diag):
        w = [p for p in pts if now - p[0] <= 1.0]
        if len(w) < 2 or w[-1][0] - w[0][0] < 0.05:
            return 0.0
        dt = w[-1][0] - w[0][0]
        return math.hypot(w[-1][1] - w[0][1], w[-1][2] - w[0][2]) / dt / diag

    @staticmethod
    def _peak(tr, now):
        return max([v for t, v in tr["speeds"] if now - t <= 2.0] or [0.0])

    def update(self, vehicles, now):
        alerts = []
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            tr = self.tracks.setdefault(v["id"], {"pts": deque(maxlen=40), "speeds": deque(maxlen=60), "seen": now, "speed": 0.0})
            tr["seen"] = now
            tr["pts"].append((now, (x1 + x2) / 2, (y1 + y2) / 2))
            tr["speed"] = self._speed(tr["pts"], now, max(1.0, math.hypot(x2 - x1, y2 - y1)))
            tr["speeds"].append((now, tr["speed"]))

        any_overlap = cfg.COLLISION_ANY_OVERLAP
        contact = cfg.COLLISION_OVERLAP if any_overlap else self.CONTACT
        confirm = cfg.COLLISION_CONFIRM_SEC if any_overlap else self.CONFIRM_SEC
        for a, b in combinations(vehicles, 2):
            if _overlap(a["box"], b["box"]) < contact or _iou(a["box"], b["box"]) > self.DUPLICATE_IOU:
                continue
            key = (min(a["id"], b["id"]), max(a["id"], b["id"]))
            ta, tb = self.tracks[a["id"]], self.tracks[b["id"]]
            pr = self.pairs.get(key)
            if pr is None:
                pr = self.pairs[key] = {"since": now, "pre": max(self._peak(ta, now), self._peak(tb, now)), "cool": 0.0}
            pr["last"] = now
            if any_overlap:      # boxes overlap for long enough -> alert, moving or not
                fire = now - pr["since"] >= confirm and now >= pr["cool"]
            else:                # strict: was moving, then both stopped
                fire = (now - pr["since"] >= confirm and pr["pre"] >= self.MOVING
                        and max(ta["speed"], tb["speed"]) < self.LOW and now >= pr["cool"])
            if fire:
                pr["cool"] = now + cfg.ALERT_COOLDOWN
                alerts.append({"type": "ROAD_ACCIDENT", "service": "AMBULANCE", "severity": "HIGH",
                               "message": f"Collision between {a['label']} #{a['id']} and {b['label']} #{b['id']}",
                               "score": round(min(1.0, 0.65 + pr["pre"]), 2),
                               "reasons": ["vehicle boxes overlap" + (" after moving" if pr["pre"] >= self.MOVING else " (vehicles static)")],
                               "signals": [{"name": "Moving fast before contact", "value": round(min(1.0, pr["pre"]), 3), "max": 1.0},
                                           {"name": "Vehicles overlap", "value": 1.0, "max": 1.0},
                                           {"name": "Both stopped after contact", "value": 1.0, "max": 1.0}],
                               "threshold": None, "sustain": confirm})

        for key in [k for k, p in self.pairs.items() if now - p.get("last", p["since"]) > 1.5]:
            del self.pairs[key]
        for vid in [k for k, t in self.tracks.items() if now - t["seen"] > 3.0]:
            del self.tracks[vid]
        return alerts
