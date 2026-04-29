# Baby Monitor - Child Safety Monitor

YOLO11-pose + facial emotion recognition + inappropriate touch detection.

## Quick Start

```bash
pip install -r requirements.txt
python detector.py
```

## Alert Types

| Alert | Trigger | Level |
|-------|---------|-------|
| `ALONE_4MIN` | 1 child alone >= 4 min | Medium |
| `CRYING_70PCT_4MIN` | Crying >= 70% of last 4-min window | High |
| `TOUCH_WARN` | Wrist near sensitive zone >= 3s | Low |
| `TOUCH_ALERT` | Wrist near private zone >= 5s OR contact + distress | High |
| `TOUCH_EMERGENCY` | Private zone >= 10s + child distress | Critical |

Two children together are NOT alone.

---

## Detection Modules

### 1. Crying — 3 independent scores + combined

| Score | Source | Weight |
|-------|--------|--------|
| Face | facial_emotion_recognition (Sad/Fear/Angry/Disgust) | 50% |
| Pose | YOLO11 keypoints — torso_curl, head_droop, arms_raised... | 35% |
| Motion | Frame-difference intensity / audio FFT | 15% |
| **Combined** | Weighted average | — |

Alert fires when combined >= 0.35 AND 4-min window ratio >= 70%.

### 2. Inappropriate Touch Detection

Body zones (from YOLO11-pose keypoints):

| Zone | Sensitivity | Keypoints |
|------|-------------|-----------|
| chest | private | left/right shoulder |
| abdomen | private | shoulder-hip midpoint |
| groin | private | left/right hip |
| inner_thigh | private | hip-knee midpoint |
| shoulder | sensitive | left/right shoulder |
| upper_arm | sensitive | shoulder-elbow |
| head | sensitive | nose, eyes |

**Per-frame algorithm:**
1. Adult wrist keypoints extracted from YOLO11-pose
2. Distance to each child zone center computed
3. Normalized by child bbox diagonal
4. Contact score = clip(1 - dist/threshold, 0, 1)
5. Duration accumulated; alerts at configurable thresholds

**False positive reduction:**
- Adult + child bboxes must physically overlap (IoU check)
- Brief contacts < warn_seconds silently ignored
- ALERT+ requires confirmed child distress (cry OR pose signals)
- Per-alert cooldown prevents notification spam

---

## Video Overlay (left panel)

```
Children:1  Adults:1
SAFE
─────────────────────
Face  [████░░]  0.68    FER emotion score
Pose  [██░░░░]  0.40    body language
Motion[█░░░░░]  0.18    motion/audio
─────────────────────
COMBINED [████░░] 0.55  << CRY
─────────────────────
4min%  [████░░]  0.61   (threshold line at 70%)
─────────────────────
Touch  [██░░░░]  0.31
  groin        2.3s  [private]
  chest        1.1s  [private]
! TOUCH ALERT !
```

---

## Key config.yaml Settings

```yaml
touch:
  proximity_threshold: 0.18   # lower = more sensitive
  warn_seconds: 3.0
  alert_seconds: 5.0
  emergency_seconds: 10.0

emotion:
  cry_emotions: ["Sad", "Fear", "Angry", "Disgust"]
  min_confidence: 0.40
  run_every_n_frames: 5

alerts:
  alone_window_seconds: 240
  cry_ratio_threshold: 0.70
```

---

## Camera Source

```yaml
source:
  type: "webcam"    # webcam | rtsp | file
  device_id: 0
  rtsp_url: "rtsp://user:pass@ip:554/stream"
  file_path: "video/test.mp4"
```

---

## Custom Model Training

```bash
python training/prepare_dataset.py --create-structure
# Label with Roboflow (child=0, adult=1)
python training/train.py
# best_custom.pt -> set model.weights in config.yaml
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| ultralytics | YOLO11-pose (detection + 17 keypoints) |
| facial-emotion-recognition | Bundled 33MB PyTorch model, offline, 95.6% acc |
| opencv-python | Video I/O + drawing |
| torch / torchvision | Emotion model backend |
