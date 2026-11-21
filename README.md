# VigiLight — AI Security Surveillance System

Real-time AI surveillance that detects threats before they escalate — not just after a weapon appears. Built with Python, YOLO11, and Flask.

---

## What It Does

VigiLight combines video, audio, and behaviour analysis across multiple camera feeds to detect threats, dispatch alerts to guards, and log evidence automatically.

| Input | AI Method | Output |
|---|---|---|
| Video (USB / IP / RTSP / file) | YOLO11n-pose + ByteTrack: people, IDs, wrist/shoulder keypoints | Per-person behaviour tracked over time |
| Video | YOLO11n + ByteTrack: vehicles, weapons (optional fine-tuned model) | Weapons linked to a person's hand; vehicle collision detection |
| Audio (IP camera mic or local mic) | YAMNet (521 AudioSet classes) or spectral fallback | Glass breaking, screaming/shouting, gunshot-like sounds |

---

## Where It Works

VigiLight is not built for one specific setup. It runs on a laptop, accepts any camera you can plug in or connect over a network, and needs no cloud service or proprietary hardware. That makes it easy to deploy for a few hours or leave running permanently.

| Scenario | How it fits |
|---|---|
| **College fests & events** | Bring a laptop and a few USB or phone cameras. Set up in minutes, tear down just as fast. Guards monitor from their phones via QR. |
| **Concerts & public gatherings** | Cover multiple entry and crowd zones with IP cameras over Wi-Fi. X-Ray mode handles bag screening at the gate. |
| **Campuses & institutions** | Permanent install with RTSP cameras already in place. Guards patrol freely — alerts follow them on their phones. |
| **Malls & venues** | Plug into existing IP camera infrastructure. No new hardware needed if cameras are already networked. |
| **Temporary checkpoints** | Any entry point that needs bag or body screening — toggle X-Ray mode on a camera pointed at a tray or conveyor. |

The system does not require internet during operation. Everything runs locally — models, evidence, alerts (except external channels like Telegram which need connectivity if configured).

---

## Features

### Behaviour-Based Threat Detection
A weapon in frame alone does not trigger an alert. Each tracked person gets a threat score (0–1) built from multiple signals over time:

- **Armed person:** weapon near hand + raised arm + fast arm motion + closing in on another person
- **Unarmed fight:** aggressive arm motion + two people converging
- **Audio fusion:** a scream, glass break, or gunshot in the last few seconds amplifies a rising visual score
- **Road accident:** two tracked vehicles overlap while moving, then stop
- Alert fires only when the score stays above the threshold for ~1 second (armed) or ~1.5 seconds (unarmed fight)

### X-Ray Scanner Mode
Toggle X-Ray mode on any camera feed to simulate a baggage or body scanner. When enabled:
- The feed switches to a high-contrast bone-colour render (CLAHE + COLORMAP_BONE) with a grid overlay
- The AI immediately flags any detected weapon or contraband with a red bounding box and confidence score
- A `CRITICAL` alert is raised for every detected object above 28% confidence — no sustained-score wait

To enable: click the **X-Ray** button on any camera card in the dashboard, or send a POST to `/api/cameras/<name>/xray`. Just plug in any supported camera and toggle the mode.

### Demo / Testing Mode
Run the full AI pipeline on pre-recorded video files without any live camera. Place `.mp4`, `.avi`, `.mkv`, `.mov`, or `.webm` files in the `videos/` folder in the project directory.

- Click **Start Demo** in the dashboard to load all videos as separate camera streams
- Select a specific video to run; the others pause automatically to save CPU
- The full detection + behaviour + alert pipeline runs exactly as it would live
- Use this to test, train staff, or demonstrate the system without a live scene

You can also upload a video directly from the dashboard (**Upload Video** tile) without putting it in the folder manually.

### Guard View (Phone Access)
Guards do not need to sit at the server. Any responsible person can monitor and respond from their phone on the same network.

1. The dashboard shows a **QR code** (generated server-side from the LAN IP)
2. Guard scans it and opens the guard view in their phone browser — no app install needed
3. The guard view shows live alerts with snapshot, severity, camera, and time
4. Guard actions: **Confirm** (real threat), **Dismiss** (false alarm), **Responding** (on my way), **Resolve** (handled)
5. Responding or confirming an alert escalates it to the external notification channels (Telegram, Discord, ntfy)
6. If a guard does not act within the verify timeout, the alert auto-escalates

Set `ACCESS_PIN` in `.env` to require a PIN before anyone can open the dashboard or guard view on the network.

### Alert Channels

Every `HIGH` or `CRITICAL` alert is stored with a snapshot and a 7-second clip (4 s before + 3 s after), and sent to whichever channels you configure:

| Channel | What is sent |
|---|---|
| **Dashboard** | Full-screen flashing overlay, siren, spoken warning, acknowledge button |
| **Telegram** | Snapshot photo with caption, then the video clip |
| **Discord** | Webhook message with snapshot, then the clip |
| **ntfy** | Push notification to the ntfy app on your phone with snapshot |
| **MQTT** | JSON payload for other systems (lighting controllers, intercoms, etc.) |

Configure any combination in `.env`. All evidence is saved locally to `evidence/`.

### Audio Detection
- **YAMNet** (Google, 521 sound classes) if TensorFlow is available
- **Spectral fallback** (zero extra dependencies) if TensorFlow is not installed — classifies glass break, scream, and gunshot from FFT band energy and crest factor
- Supported sources: IP camera audio stream (auto-detected from the camera URL), local microphone (`mic` or `mic:2`), or any HTTP WAV stream

### Sensitivity Presets
Switch between **Low / Medium / High** sensitivity from the dashboard Settings panel without restarting. Changes take effect immediately on the next processed frame.

---

## Running It

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in alert channels you want
python app.py               # open http://localhost:5000
```

Add cameras from the dashboard (USB / IP / RTSP / video file, **Scan USB** to find webcams), or pass them on the command line:

```bash
python app.py --camera 0 --camera 192.168.1.20:8080 --camera rtsp://user:pw@host/stream
```

**IP Webcam (Android):** type `192.168.1.20:8080` — `/video` and `/audio.wav` are added automatically.  
**USB webcam:** select *Local microphone* for audio.  
**Test your setup:** click **Send test alert** to fire a fake CRITICAL alert through every configured channel.

The first run downloads YOLO weights automatically. The feed appears after a few seconds.

---

## Training a Weapon Model (Optional)

COCO's knife class is weak and has no gun class. Train a small custom detector on your own dataset to improve accuracy.

1. **Get a dataset** in YOLO format with `gun` and `knife` classes. [Roboflow Universe](https://universe.roboflow.com) has public weapon-detection datasets — search "gun knife detection" and export as *YOLOv11*. Check each dataset's licence and make sure it includes realistic CCTV-style examples.

2. **Train:**
   ```bash
   python train_weapons.py --data path/to/data.yaml
   ```
   Trains YOLO11s at 960 px, prints precision / recall / mAP, and copies the best weights to `weights/weapons.pt`. A free Colab GPU takes roughly 30–60 minutes.

3. **Restart VigiLight.** It loads `weights/weapons.pt` automatically. Every class in the model is treated as a weapon; COCO's knife/scissors/bat classes are disabled unless `WEAPON_KEEP_COCO=true`.

Use `--base yolo11n.pt` if speed is a concern. Lower `PROCESS_FPS` or `WEAPON_IMGSZ` in `.env` if the AI fps drops after adding the weapon model.

---

## Evaluation

```bash
python evaluate.py path/to/video.mp4 --out results.csv
```

Runs the full detection + behaviour pipeline on a recorded clip and prints every alert with its video timestamp and score. Use this to build a results table: record staged scenarios (calm knife use, staged attack, fight, normal walking, a fender-bender), run evaluate on each, and compare alerts against what actually happened.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `THREAT_THRESHOLD` | `0.60` | Score above which an alert fires |
| `WARNING_THRESHOLD` | `0.35` | Score shown as a warning on the dashboard |
| `PROCESS_FPS` | `8` | Frames per second fed to the AI |
| `WEAPON_MODEL` | _(auto)_ | Path to a custom YOLO weapon model |
| `WEAPON_IMGSZ` | `960` | Input resolution for the weapon model |
| `AUDIO_ENABLED` | `true` | Enable/disable audio analysis |
| `AUDIO_THRESHOLD` | `0.5` | Minimum score to register an audio event |
| `SAVE_CLIPS` | `true` | Save 7-second video clips as evidence |
| `ACCESS_PIN` | _(none)_ | Optional PIN for dashboard and guard view |
| `VERIFY_FIRST` | `false` | Hold external alerts until a guard confirms |
| `VERIFY_TIMEOUT` | `45` | Auto-escalate after this many seconds if no guard response |
| `DEMO_NOTIFY` | `false` | Send demo alerts to external channels |
| `DEVICE` | _(auto)_ | `0` for first GPU, `cpu` to force CPU |

---

## Limitations

- Scores *behaviour that looks like an attack* — not intent. Sparring, dance, or sports can resemble aggression.
- Pose and weapon detection need a reasonably clear, well-lit view. Small knives far from the camera are often missed.
- Fireworks and similar impulses can resemble gunshots to the audio model.
- Designed to alert a human who decides what to do — not to act autonomously.
- Camera footage is personal data. Use only where permitted, keep `evidence/` private, and never commit `.env`.
