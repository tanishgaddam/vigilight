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

## Dashboard

- **Header stats:** cameras online, active threats and open alerts at a glance; the browser tab title also shows `[ALERT]` or the open count.
- **Camera cards:** colour-coded level chip, threat meter with the warning/alert thresholds marked, people count, camera and AI fps, audio status. Hover a card for focus view, fullscreen and remove. Layout switch: Auto / 1 / 2 / 3 columns.
- **Incident log:** filter by Open / Critical / High and by camera, click a snapshot for the incident detail (clip player, reasons, confidence), **Acknowledge all**.
- **Alarm overlay:** flashing full-screen alert with snapshot, siren, spoken warning; **Sound** must be enabled once per page load (browser rule).
- **Add camera:** USB / IP / RTSP / video-file tiles with hints and **Scan USB**.
- Shows a banner when the server connection drops; keyboard friendly (Esc closes dialogs).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add the alert channels you want
python app.py               # then open http://localhost:5000
```

The feed shows `CAM n fps` (what the camera delivers) and `AI n fps` (how often detection runs) in the top-left corner.

Add cameras in the dashboard (USB / IP / RTSP / video file, **Scan USB** finds webcams), or:

```bash
python app.py --camera 0 --camera 192.168.1.20:8080 --camera rtsp://user:pw@192.168.1.30/stream
```

- **IP Webcam style phone apps:** type `192.168.1.20:8080`; `/video` and `/audio.wav` are added automatically.
- **USB webcam:** pick *Local microphone* for audio if you want sound analysis.
- **Test your setup:** the **Send test alert** button fires a fake CRITICAL alert through every configured channel.
- The first start downloads YOLO and YAMNet weights; the feed appears after a few seconds.

## Train the knife and gun model

COCO's knife class is weak and it has no gun class, so train a small detector on your own data.

1. **Get a dataset in YOLO format** with at least `gun` and `knife` classes. Roboflow Universe has public
   weapon-detection datasets: search "gun knife detection" and export as *YOLOv11* (it gives you a folder with `data.yaml`).
   Open Images also has Handgun and Knife classes. Check each dataset's licence, and make sure it includes
   realistic CCTV-style, small and partly hidden examples, not only close-up stock photos.
2. **Train** (free Colab GPU: Runtime -> Change runtime type -> GPU, upload the project, `pip install ultralytics`):
   ```bash
   python train_weapons.py --data path/to/data.yaml
   ```
   It trains YOLO11s at 960 px, prints precision / recall / mAP, and copies the best weights to `weights/weapons.pt`.
3. **Restart VigiLight.** It loads `weights/weapons.pt` automatically (or set `WEAPON_MODEL=` in `.env`).
   Every class in that model counts as a weapon, and COCO knife/scissors/bat are switched off unless `WEAPON_KEEP_COCO=true`.

Put the mAP / precision / recall numbers in your report. Use `yolo11n.pt` as `--base` if the speed is too low on your PC.
The weapon model adds a third network per frame, so lower `PROCESS_FPS` or `WEAPON_IMGSZ` if the AI fps drops.

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
