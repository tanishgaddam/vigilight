# VigiLight - AI street-light safety monitor

Smart street-light cameras that raise an alert when a situation is *developing into
violence*, not merely when a knife appears in frame. Python, YOLO11 (detection, pose,
tracking), YAMNet (audio events), Flask dashboard, multi-channel alerts.

## What it does

| Input | AI method | Result |
|---|---|---|
| Video (USB, IP/phone, RTSP, file) | YOLO11n-pose + ByteTrack: people, IDs, wrist/shoulder keypoints | per-person behaviour tracked over time |
| Video | YOLO11n + ByteTrack: vehicles, knife / scissors / bat (optional fine-tuned weapon model) | weapons linked to a person's hand, vehicle tracks |
| Audio (IP-camera mic or local mic) | YAMNet (521 sound classes) | glass breaking, screaming/shouting, gunshot-like sounds |

### Attack-risk logic (not "knife = alarm")
Every tracked person gets a threat score (0-1), smoothed over time:

- **Armed person:** weapon at the hand + arm raised + fast arm motion + close to / closing in on another person
- **Unarmed fight:** fast arm motion next to another person, the other person moving aggressively too
- **Audio fusion:** a scream / glass / gunshot in the last few seconds adds to a visual score that is already rising
- **Levels:** NORMAL -> WATCH (weapon visible, nothing aggressive: log only) -> WARNING (dashboard only) -> THREAT
- An alert fires only when the score stays above the threshold for about 1 second (armed) or 1.5 seconds (fight)
- **Road accidents:** two tracked vehicles overlap while moving, then both stop

Audio alone can also raise glass-break, gunshot and scream alerts.

### Alerts
Dashboard (full-screen flashing overlay, siren, spoken warning, acknowledge button), Telegram
(snapshot + clip), Discord webhook, ntfy phone push, MQTT. Every alert stores a snapshot and a
7-second clip (4 s before, 3 s after) in `evidence/`.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add the alert channels you want
python app.py               # then open http://localhost:5000
```

Add cameras in the dashboard (USB / IP / RTSP / video file, **Scan USB** finds webcams), or:

```bash
python app.py --camera 0 --camera 192.168.1.20:8080 --camera rtsp://user:pw@192.168.1.30/stream
```

- **IP Webcam style phone apps:** type `192.168.1.20:8080`; `/video` and `/audio.wav` are added automatically.
- **USB webcam:** pick *Local microphone* for audio if you want sound analysis.
- **Test your setup:** the **Send test alert** button fires a fake CRITICAL alert through every configured channel.
- The first start downloads YOLO and YAMNet weights; the feed appears after a few seconds.

## Evaluate (for your results table)

```bash
python evaluate.py clips/staged_attack.mp4 --out results.csv
```

Record staged clips (calm knife use, staged attack, staged fight, normal walking, a fender-bender with toy cars)
and compare the alerts and timestamps with what really happened: true alerts, false alarms, missed events, delay.

## Tuning (`.env`)

`THREAT_THRESHOLD`, `WARNING_THRESHOLD`, `STRIKE_SPEED`, `SUSTAIN_ARMED`, `AUDIO_THRESHOLD`, `PROCESS_FPS`.
Raise the threshold if you get false alarms, lower it if staged attacks are missed. `WEAPON_MODEL=path.pt`
plugs in a YOLO model fine-tuned on guns and knives, which helps a lot because COCO's knife class is weak.

## Limits (be upfront about these)

- It scores *behaviour that looks like an attack*; it cannot read intent. Sparring, dancing and sports can look aggressive.
- Pose and weapon detection need a reasonably clear, well-lit view; small knives far from the camera are often missed.
- Fireworks and similar bangs can resemble gunshots to the audio model.
- Designed to alert a human who decides what to do, not to act on its own.
- Camera footage is personal data: use it only where you are allowed to, keep `evidence/` private, and never commit `.env`.
