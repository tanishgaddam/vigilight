"""Run the full detection + behaviour pipeline on a recorded video and list the alerts
with video timestamps. Use it to build the results table for your hackathon report:
record staged clips (calm knife use, a staged attack, a staged fight, normal walking...),
run this on each, and compare the alerts against what you know happened.

    python evaluate.py clips/staged_attack.mp4 --out results.csv
"""
import argparse
import csv

import cv2

import config as cfg
from behavior import CollisionEngine, ThreatEngine
from detector import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=cfg.PROCESS_FPS, help="analysis rate")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native / args.fps))
    detector, threat, collision = Detector(), ThreatEngine(), CollisionEngine()
    events, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / native  # video time drives the engines, so results are repeatable
            det = detector.detect(frame)
            _, alerts = threat.update(det["persons"], det["weapons"], None, t)
            alerts += collision.update(det["vehicles"], t)
            for a in alerts:
                events.append((round(t, 2), a["type"], a["score"], "; ".join(a["reasons"])))
                print(f"{t:7.2f}s  {a['type']:<18} score={a['score']:.2f}  {'; '.join(a['reasons'])}")
        idx += 1
    print(f"\n{len(events)} alert(s) in {idx / native:.1f}s of video")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_time_s", "type", "score", "reasons"])
            w.writerows(events)


if __name__ == "__main__":
    main()
