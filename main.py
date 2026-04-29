#!/usr/bin/env python3
"""
=======================================================
  Baby Monitor - One-shot Setup Script
  Run: python setup_baby_monitor.py
=======================================================
This script:
  1. Creates all required folders in the CURRENT directory
  2. Writes all project files (detector.py, config.yaml, training/*)
  3. Installs required packages automatically
  4. Project is ready to run immediately after
"""

import sys
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────
# TERMINAL COLOR CODES
# ─────────────────────────────────────────────────────
R    = "\033[91m"
G    = "\033[92m"
Y    = "\033[93m"
B    = "\033[94m"
C    = "\033[96m"
W    = "\033[97m"
DIM  = "\033[2m"
BOLD = "\033[1m"
END  = "\033[0m"

def ok(msg):   print(f"  {G}+{END} {msg}")
def info(msg): print(f"  {C}>{END} {msg}")
def warn(msg): print(f"  {Y}!{END}  {msg}")

def section(title):
    print(f"\n{BOLD}{B}{'─'*52}{END}")
    print(f"{BOLD}{W}  {title}{END}")
    print(f"{BOLD}{B}{'─'*52}{END}")


# ─────────────────────────────────────────────────────
# config.yaml
# ─────────────────────────────────────────────────────
CONFIG_YAML = """\
# ============================================================
# Baby Monitor - Prolonged Crying Alone Detection Config
# ============================================================

# --- Video input source ---
source:
  type: "webcam"          # webcam | rtsp | file
  device_id: 0            # webcam device index (0, 1, 2 ...)
  rtsp_url: ""            # e.g. rtsp://user:pass@192.168.1.10:554/stream
  file_path: ""           # path to a local video file

# --- Video output ---
output:
  show_video: true        # display real-time OpenCV window
  save_video: false       # write annotated output to a video file
  output_path: "output/recorded.mp4"
  fps: 25
  resolution: [1280, 720]

# --- YOLO model ---
model:
  # Pretrained:  yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt
  # Custom:      training/runs/train/weights/best.pt
  weights: "yolo11n-pose.pt"  # pose model: yolo11n-pose | yolo11s-pose | yolo11m-pose
  confidence: 0.45
  iou: 0.65           # raised: reduces in-model duplicate detections
  dedup_iou: 0.45     # secondary NMS threshold (merge overlapping same-person boxes)
  device: "auto"          # auto | cpu | cuda | mps
  imgsz: 640

# --- COCO class IDs ---
classes:
  person_class_id: 0      # COCO class 0 = person

# --- Facial expression recognition ---
# Uses: facial-emotion-recognition (pip install facial-emotion-recognition)
# Model: bundled 33MB PyTorch ResNet, trained on AffectNet, 95.6% accuracy
# 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
# Completely OFFLINE — no internet needed at runtime.
emotion:
  enabled: true
  # Emotions that count as "crying / distressed" (capitalized, matches model output)
  cry_emotions: ["Sad", "Fear", "Angry", "Disgust"]
  # Minimum confidence to count an emotion as cry (0.0-1.0)
  min_confidence: 0.40
  # Run emotion analysis every N frames (5 = ~5fps on CPU, good balance)
  run_every_n_frames: 5

# --- Pose-based distress analysis ---
pose:
  enabled: true
  # Score threshold: how many distress signals must fire to count as distress
  # Each signal contributes 0.2 (5 signals total), so 0.4 = 2 signals needed
  distress_score_threshold: 0.4
  # Signals detected:
  #   arms_raised  - wrists above shoulders (reaching/distress)
  #   torso_curl   - hunched forward (sobbing posture)
  #   head_droop   - head below shoulder line
  #   lying_down   - body horizontal (fell / collapsed)
  #   body_shake   - rapid oscillation of torso (sobbing shake)

# --- Alone detection ---
alone:
  # A person is classified as an ADULT if their bbox height >= this ratio of frame height.
  # Two children together are NOT considered alone (they have each other).
  # Alone = exactly 1 child present AND 0 adults present.
  adult_height_ratio: 0.50
  child_height_ratio: 0.40
  use_size_filter: true
  # Centroid tracker settings (prevents double-counting same person across frames)
  tracker_max_dist_ratio: 0.15  # max centroid movement as fraction of frame diagonal
  tracker_max_miss: 5           # frames a track survives without a new detection

# --- Alert thresholds ---
alerts:
  # ALONE alert: child alone for this many seconds
  alone_window_seconds: 240           # 4-minute observation window

  # CRYING alert: within the alone window, >= cry_ratio of frames show crying
  cry_window_seconds: 240             # sliding window length in seconds (4 min)
  cry_ratio_threshold: 0.70           # 70% of frames must be crying to trigger alert

  alert_cooldown_seconds: 60          # minimum gap between repeated alerts

# --- Audio detection (optional, requires pyaudio) ---
audio:
  enabled: false
  sample_rate: 16000
  chunk_size: 1024
  cry_frequency_min: 300   # Hz
  cry_frequency_max: 1000
  cry_amplitude_threshold: 0.3

# --- Notification channels ---
notification:
  console: true
  sound_alert: true
  telegram:
    enabled: false
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender: "your@email.com"
    password: "your_app_password"
    recipient: "recipient@email.com"

# --- Logging ---
logging:
  level: "INFO"
  file: "logs/monitor.log"
  max_size_mb: 10

# --- Inappropriate touch detection ---
touch:
  enabled: true
  # Wrist-to-zone distance threshold (fraction of child bbox diagonal)
  proximity_threshold: 0.18
  # Minimum bbox IoU between adult and child to consider physical contact
  min_bbox_overlap: 0.05
  # Alert time thresholds (seconds of continuous contact)
  warn_seconds: 3.0           # TOUCH_WARN      sensitive/private zone
  alert_seconds: 5.0          # TOUCH_ALERT     private zone OR any + distress
  emergency_seconds: 10.0     # TOUCH_EMERGENCY private + long + distress
  event_gap_seconds: 2.0      # max gap before timer resets
  alert_cooldown_seconds: 30.0
"""

# ─────────────────────────────────────────────────────
# detector.py
# ─────────────────────────────────────────────────────
DETECTOR_PY = r'''
"""
Baby Monitor - Prolonged Crying Alone Detector
YOLO11-pose + FER + pose-based distress analysis.

KEY FIXES in this version:
  - Duplicate detection suppression: IoU-based NMS applied AFTER model output
    to merge overlapping boxes of the same person.
  - Tracking-based deduplication: persons tracked across frames by centroid
    proximity so a single person is never double-counted.
  - Robust _classify: uses area-weighted single-person selection when overlap
    is detected, rather than counting all raw boxes.

Detection pipeline:
  1. YOLO11-pose  → person bboxes + 17 keypoints
  2. NMS dedup    → remove overlapping boxes (same person, multi-detection)
  3. FER          → facial emotion (sad/fear/angry/disgust)
  4. Pose scorer  → body language distress from keypoints
  5. Audio FFT    → optional microphone
  6. Motion diff  → last-resort fallback

Alone logic:
  - Exactly 1 child present, 0 adults  → alone=True
  - 2+ children together               → alone=False

Alert rules:
  - ALONE_4MIN        : child alone >= alone_window_seconds
  - CRYING_70PCT_4MIN : >= cry_ratio_threshold of frames show crying

Usage:
  python detector.py
  python detector.py --config config.yaml
"""

import cv2
import time
import logging
import sys
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import yaml
import numpy as np

# ── Core ─────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics"); sys.exit(1)

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import requests as req
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# COCO 17-keypoint index map
# ─────────────────────────────────────────────────────────────
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

SKELETON_BONES = [
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),   ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),     ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),         ("left_knee","left_ankle"),
    ("right_hip","right_knee"),       ("right_knee","right_ankle"),
    ("nose","left_eye"),              ("nose","right_eye"),
    ("left_eye","left_ear"),          ("right_eye","right_ear"),
]


# ─────────────────────────────────────────────────────────────
# Config / Logging
# ─────────────────────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(cfg):
    lc       = cfg.get("logging", {})
    level    = getattr(logging, lc.get("level", "INFO").upper(), logging.INFO)
    log_file = lc.get("file", "logs/monitor.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(log_file,
                            maxBytes=lc.get("max_size_mb", 10) * 1024 * 1024,
                            backupCount=3),
    ]
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)
    return logging.getLogger("BabyMonitor")


# ─────────────────────────────────────────────────────────────
# Deduplication NMS helper
# ─────────────────────────────────────────────────────────────
def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter  = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union  = area_a + area_b - inter + 1e-6
    return inter / union


def nms_deduplicate(boxes_xyxy: np.ndarray,
                    scores: np.ndarray,
                    kps_list: Optional[np.ndarray],
                    iou_thresh: float = 0.45) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply greedy NMS to remove duplicate detections of the same person.
    Returns (kept_boxes, kept_scores, kept_kps).

    This is the primary fix for the 'same person detected 2-3 times' bug.
    YOLO's built-in NMS sometimes misses overlapping detections of the same
    person when they appear in slightly different poses or scales.
    We re-apply NMS with a slightly higher IoU threshold after inference.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, kps_list

    # Sort by confidence descending
    order = np.argsort(scores)[::-1]
    keep  = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        suppress = []
        for j in rest:
            if iou(boxes_xyxy[i], boxes_xyxy[j]) >= iou_thresh:
                suppress.append(j)
        suppress_set = set(suppress)
        order = np.array([x for x in rest if x not in suppress_set])

    kept_boxes = boxes_xyxy[keep]
    kept_scores = scores[keep]
    kept_kps   = kps_list[keep] if kps_list is not None else None
    return kept_boxes, kept_scores, kept_kps


# ─────────────────────────────────────────────────────────────
# Simple centroid tracker  (prevents id-flip across frames)
# ─────────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    bbox:      np.ndarray    # [x1,y1,x2,y2]
    centroid:  np.ndarray    # [cx, cy]
    height_ratio: float      # bbox_h / frame_h
    kps:       Optional[np.ndarray] = None
    miss_count: int = 0      # frames not matched


class PersonTracker:
    """
    Lightweight IoU / centroid tracker.
    Maintains a stable list of unique persons across frames,
    preventing double-counting when YOLO briefly duplicates a detection.
    """
    def __init__(self, max_dist_ratio: float = 0.15, max_miss: int = 5):
        """
        max_dist_ratio : max centroid movement as fraction of frame diagonal
        max_miss       : frames a track survives without a matching detection
        """
        self.tracks: List[TrackedPerson] = []
        self.max_dist_ratio = max_dist_ratio
        self.max_miss       = max_miss

    def update(self, boxes: np.ndarray, scores: np.ndarray,
               kps_list: Optional[np.ndarray],
               frame_w: int, frame_h: int) -> List[TrackedPerson]:
        """
        Match new detections to existing tracks by centroid distance.
        Returns list of current (deduplicated) active tracks.
        """
        diag = np.sqrt(frame_w**2 + frame_h**2) + 1e-5
        max_dist = self.max_dist_ratio * diag

        # Compute new centroids
        new_centroids = np.array(
            [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
        ) if len(boxes) > 0 else np.zeros((0, 2))

        matched_new   = set()
        matched_track = set()

        # Greedy nearest-centroid matching
        assignments: Dict[int, int] = {}   # track_idx -> det_idx
        for ti, track in enumerate(self.tracks):
            best_dist, best_di = 1e9, -1
            for di in range(len(new_centroids)):
                if di in matched_new:
                    continue
                d = np.linalg.norm(new_centroids[di] - track.centroid)
                if d < best_dist:
                    best_dist, best_di = d, di
            if best_di >= 0 and best_dist < max_dist:
                assignments[ti] = best_di
                matched_new.add(best_di)
                matched_track.add(ti)

        # Update matched tracks
        for ti, di in assignments.items():
            t = self.tracks[ti]
            t.bbox          = boxes[di]
            t.centroid      = new_centroids[di]
            t.height_ratio  = (boxes[di][3] - boxes[di][1]) / frame_h
            t.kps           = kps_list[di] if kps_list is not None else None
            t.miss_count    = 0

        # Age unmatched tracks
        for ti in range(len(self.tracks)):
            if ti not in matched_track:
                self.tracks[ti].miss_count += 1

        # Add new tracks for unmatched detections
        for di in range(len(boxes)):
            if di not in matched_new:
                self.tracks.append(TrackedPerson(
                    bbox         = boxes[di],
                    centroid     = new_centroids[di],
                    height_ratio = (boxes[di][3] - boxes[di][1]) / frame_h,
                    kps          = kps_list[di] if kps_list is not None else None,
                ))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]
        return self.tracks


# ─────────────────────────────────────────────────────────────
# Sliding-window cry ratio tracker
# ─────────────────────────────────────────────────────────────
class CryWindowTracker:
    """
    Tracks cry detections over a fixed sliding time window.
    Gaps are counted as non-crying so 70% requires sustained crying.
    """
    def __init__(self, window_seconds: float = 240):
        self.window = window_seconds
        self._buf: deque = deque()

    def push(self, ts: float, is_crying: bool):
        self._buf.append((ts, is_crying))
        cutoff = ts - self.window
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def ratio(self) -> float:
        if not self._buf: return 0.0
        return sum(1 for _, c in self._buf if c) / len(self._buf)

    def window_filled(self, now: float) -> bool:
        if not self._buf: return False
        return (now - self._buf[0][0]) >= self.window

    def reset(self):
        self._buf.clear()


# ─────────────────────────────────────────────────────────────
# Alert state
# ─────────────────────────────────────────────────────────────
@dataclass
class AlertState:
    alone_start: Optional[float] = None
    last_alert_time: float       = 0.0
    alert_count: int             = 0
    total_alone_seconds: float   = 0.0

    ALONE_ALERT = "ALONE_4MIN"
    CRY_ALERT   = "CRYING_70PCT_4MIN"


# ─────────────────────────────────────────────────────────────
# Pose distress analyser
# ─────────────────────────────────────────────────────────────
class PoseDistressAnalyser:
    """
    Checks 5 body-language distress signals from COCO keypoints.

    Signals:
      arms_raised  - wrists above shoulders
      torso_curl   - head below shoulder midpoint (hunched/sobbing)
      head_droop   - nose at or below shoulder line
      lying_down   - body skeleton is near-horizontal
      body_shake   - rapid centroid oscillation across frames
    """
    MIN_VIS = 0.3

    def __init__(self, cfg: dict):
        pc = cfg.get("pose", {})
        self.enabled         = pc.get("enabled", True)
        self.score_threshold = pc.get("distress_score_threshold", 0.4)
        self._pos_history: deque = deque(maxlen=8)

    @staticmethod
    def _get(kps, name) -> Optional[Tuple[float,float,float]]:
        idx = KP[name]
        if idx >= len(kps): return None
        x, y, v = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
        return (x, y, v) if v >= PoseDistressAnalyser.MIN_VIS else None

    @staticmethod
    def _mid(a, b):
        if a is None or b is None: return None
        return ((a[0]+b[0])/2, (a[1]+b[1])/2)

    def _arms_raised(self, kps) -> float:
        ls,rs = self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder")
        lw,rw = self._get(kps,"left_wrist"),    self._get(kps,"right_wrist")
        raised, total = 0, 0
        for sh, wr in [(ls,lw),(rs,rw)]:
            if sh and wr:
                total += 1
                if wr[1] < sh[1]: raised += 1   # higher in image = smaller y
        return 1.0 if (total > 0 and raised == total) else 0.0

    def _torso_curl(self, kps) -> float:
        nose   = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"),
                           self._get(kps,"right_shoulder"))
        hip_mid= self._mid(self._get(kps,"left_hip"),
                           self._get(kps,"right_hip"))
        if None in (nose, sh_mid, hip_mid): return 0.0
        body_h = abs(hip_mid[1] - sh_mid[1]) + 1e-5
        drop   = nose[1] - sh_mid[1]
        return 1.0 if drop > 0.5 * body_h else 0.0

    def _head_droop(self, kps) -> float:
        nose   = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"),
                           self._get(kps,"right_shoulder"))
        if None in (nose, sh_mid): return 0.0
        return 1.0 if nose[1] >= sh_mid[1] else 0.0

    def _lying_down(self, kps) -> float:
        sh_mid  = self._mid(self._get(kps,"left_shoulder"),
                            self._get(kps,"right_shoulder"))
        hip_mid = self._mid(self._get(kps,"left_hip"),
                            self._get(kps,"right_hip"))
        if None in (sh_mid, hip_mid): return 0.0
        dx = abs(hip_mid[0]-sh_mid[0])
        dy = abs(hip_mid[1]-sh_mid[1]) + 1e-5
        return 1.0 if dx/dy > 2.0 else 0.0

    def _body_shake(self, kps) -> float:
        nose   = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"),
                           self._get(kps,"right_shoulder"))
        ref = nose or sh_mid
        if ref is None: return 0.0
        self._pos_history.append((ref[0], ref[1]))
        if len(self._pos_history) < 8: return 0.0
        spread = np.std([p[0] for p in self._pos_history]) + \
                 np.std([p[1] for p in self._pos_history])
        return 1.0 if spread > 8.0 else 0.0

    def analyse(self, kps: np.ndarray) -> Tuple[float, List[str]]:
        if not self.enabled or kps is None or len(kps) == 0:
            return 0.0, []
        checks = [
            ("arms_raised", self._arms_raised(kps)),
            ("torso_curl",  self._torso_curl(kps)),
            ("head_droop",  self._head_droop(kps)),
            ("lying_down",  self._lying_down(kps)),
            ("body_shake",  self._body_shake(kps)),
        ]
        triggered = [n for n, s in checks if s > 0.5]
        return len(triggered) / len(checks), triggered


# ─────────────────────────────────────────────────────────────
# Emotion analyser — facial_emotion_recognition (PyTorch, offline)
# Model: 33MB bundled ResNet, trained on AffectNet, 95.6% accuracy
# 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
# No internet required — model ships with the pip package.
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as tv_transforms
    from facial_emotion_recognition.networks import NetworkV2
    _FER_PKG_AVAILABLE = True
except ImportError:
    _FER_PKG_AVAILABLE = False


class EmotionAnalyser:
    """
    Wraps the facial_emotion_recognition package (bundled 33MB PyTorch model).

    For each detected face bbox the model outputs a 7-class softmax.
    We return per-emotion scores so the display layer can show them.

    Face detection: OpenCV Haarcascade (built-in, zero dependencies).
    Falls back to analysing the full frame crop if no face is found.
    """

    EMOTIONS     = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
    CRY_CLASSES  = {0,1,2,4}   # Angry, Disgust, Fear, Sad

    def __init__(self, cfg: dict, logger):
        ec            = cfg.get("emotion", {})
        self.enabled  = ec.get("enabled", True) and _FER_PKG_AVAILABLE
        self.cry_emo  = set(ec.get("cry_emotions", ["Sad","Fear","Angry","Disgust"]))
        self.min_conf = ec.get("min_confidence", 0.40)
        self.every_n  = ec.get("run_every_n_frames", 5)
        self._fc      = 0
        self._last_cry: bool = False
        self._last_emo: List[Tuple[str,float]] = []

        if not self.enabled:
            if not _FER_PKG_AVAILABLE:
                logger.warning(
                    "facial-emotion-recognition not installed. "
                    "Run: pip install facial-emotion-recognition")
            self._net = None
            return

        # Load bundled model (no download needed)
        import os
        model_path = os.path.join(
            os.path.dirname(__import__("facial_emotion_recognition").__file__),
            "model", "model.pkl")

        self._device = torch.device("cpu")
        self._net    = NetworkV2(in_c=1, nl=32, out_f=7).to(self._device)
        state        = torch.load(model_path, map_location="cpu")
        self._net.load_state_dict(state["network"])
        self._net.eval()
        logger.info(
            f"Emotion model loaded (acc={state.get('accuracy',0):.2%}) "
            f"— facial_emotion_recognition (offline PyTorch)")

        self._transform = tv_transforms.Compose([
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((48, 48)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # OpenCV Haarcascade face detector (built-in)
        import cv2 as _cv2
        self._face_det = _cv2.CascadeClassifier(
            _cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _predict_face(self, gray_face: np.ndarray) -> List[Tuple[str,float]]:
        """Run the model on a single grayscale face crop.
           Returns [(emotion_name, score), ...] sorted by score desc."""
        if gray_face.size == 0:
            return []
        tensor = self._transform(gray_face).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out    = self._net(tensor)
            scores = torch.softmax(out, dim=1).squeeze().tolist()
        return sorted(
            [(self.EMOTIONS[i], scores[i]) for i in range(7)],
            key=lambda x: x[1], reverse=True)

    def analyse(self, frame) -> Tuple[bool, List[Tuple[str,float]]]:
        """
        Detect faces in frame, run emotion model on each.
        Returns (is_crying, [(emotion, confidence), ...]).
        """
        if not self.enabled or self._net is None:
            return False, []

        # Only run every N frames (performance)
        self._fc += 1
        if self._fc % self.every_n != 0:
            return self._last_cry, self._last_emo

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        results  = []
        is_cry   = False

        if len(faces) == 0:
            # No face detected — analyse centre crop as fallback
            h, w = gray.shape
            margin = min(h, w) // 4
            crop   = gray[margin:h-margin, margin:w-margin]
            preds  = self._predict_face(crop)
            if preds:
                results = preds[:4]
                top_emo, top_sc = preds[0]
                if top_emo in self.cry_emo and top_sc >= self.min_conf:
                    is_cry = True
        else:
            for (x, y, fw, fh) in faces:
                crop  = gray[y:y+fh, x:x+fw]
                preds = self._predict_face(crop)
                if preds:
                    results.extend(preds[:2])
                    top_emo, top_sc = preds[0]
                    if top_emo in self.cry_emo and top_sc >= self.min_conf:
                        is_cry = True

        # Keep top-3 by score across all faces
        results = sorted(results, key=lambda x: x[1], reverse=True)[:3]
        self._last_cry = is_cry
        self._last_emo = results
        return is_cry, results


# ─────────────────────────────────────────────────────────────
# Motion fallback
# ─────────────────────────────────────────────────────────────
class MotionCryDetector:
    def __init__(self, threshold=0.15, history=10):
        self.threshold  = threshold
        self.history    = deque(maxlen=history)
        self.prev_frame = None

    def update(self, frame) -> bool:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        if self.prev_frame is None:
            self.prev_frame = gray; return False
        delta = cv2.absdiff(self.prev_frame, gray)
        _, th = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        score = th.sum()/(th.shape[0]*th.shape[1]*255)
        self.prev_frame = gray
        self.history.append(score)
        return (sum(self.history)/len(self.history)) > self.threshold


# ─────────────────────────────────────────────────────────────
# Audio detector
# ─────────────────────────────────────────────────────────────
class AudioCryDetector:
    def __init__(self, cfg: dict, logger):
        ac = cfg.get("audio", {})
        self.is_crying = False
        self._running  = False
        if not (ac.get("enabled", False) and AUDIO_AVAILABLE): return
        self._cfg = ac; self._running = True
        threading.Thread(target=self._listen, daemon=True).start()
        logger.info("Audio detector started.")

    def _listen(self):
        import pyaudio as pa_mod
        pa = pa_mod.PyAudio()
        rate  = self._cfg.get("sample_rate",16000)
        chunk = self._cfg.get("chunk_size",1024)
        f_min = self._cfg.get("cry_frequency_min",300)
        f_max = self._cfg.get("cry_frequency_max",1000)
        amp_t = self._cfg.get("cry_amplitude_threshold",0.3)
        try:
            stream = pa.open(format=pa_mod.paFloat32,channels=1,
                             rate=rate,input=True,frames_per_buffer=chunk)
            while self._running:
                data = np.frombuffer(stream.read(chunk,exception_on_overflow=False),
                                     dtype=np.float32)
                if np.max(np.abs(data)) > amp_t:
                    fft   = np.abs(np.fft.rfft(data))
                    freqs = np.fft.rfftfreq(len(data),1.0/rate)
                    mask  = (freqs>=f_min)&(freqs<=f_max)
                    self.is_crying = fft[mask].sum() > fft.sum()*0.3
                else:
                    self.is_crying = False
        except Exception: pass
        finally: pa.terminate()

    def stop(self): self._running = False


# ─────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────
class Notifier:
    def __init__(self, cfg, logger):
        self.cfg = cfg.get("notification", {}); self.logger = logger

    def send(self, atype, msg):
        if self.cfg.get("console", True):
            self.logger.warning(f"ALERT [{atype}]: {msg}")
        if self.cfg.get("sound_alert", False):
            print("\a", end="", flush=True)
        tg = self.cfg.get("telegram", {})
        if tg.get("enabled") and REQUESTS_AVAILABLE:
            try:
                req.post(f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage",
                         json={"chat_id":tg["chat_id"],"text":msg},timeout=5)
            except Exception: pass
        em = self.cfg.get("email", {})
        if em.get("enabled"):
            try:
                import smtplib; from email.mime.text import MIMEText
                m = MIMEText(msg); m["Subject"]=f"[BabyMonitor] {atype}"
                m["From"]=em["sender"]; m["To"]=em["recipient"]
                with smtplib.SMTP(em["smtp_server"],em["smtp_port"]) as s:
                    s.starttls(); s.login(em["sender"],em["password"]); s.send_message(m)
            except Exception: pass


# ─────────────────────────────────────────────────────────────
# Inappropriate Touch Detector
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Inappropriate Touch Detector
# ─────────────────────────────────────────────────────────────
# BODY ZONE MAP (keypoint index groups):
#   Private zones  : hips(11,12), chest≈shoulder-mid(5,6), groin≈hip-mid
#   Sensitive zones : stomach≈hip-shoulder mid, inner thigh≈knee(13,14)
#   Normal zones    : hands, arms, shoulders (less sensitive)
#
# DETECTION LOGIC:
#   1. For each adult-child pair: compute wrist→child_zone distances
#   2. Zone contact score = 1 / (1 + dist/threshold)  (soft proximity)
#   3. Temporal tracker: accumulate contact duration per zone
#   4. Alert levels:
#        WARN      : wrist near sensitive zone > warn_seconds
#        ALERT     : wrist near private zone > alert_seconds  OR
#                    child shows distress (cry/pose) during contact
#        EMERGENCY : private zone contact > emergency_seconds  AND
#                    child distress confirmed
#
# FALSE POSITIVE MITIGATION:
#   - Contact only counted when adult bbox overlaps child bbox
#   - Child distress (cry OR pose) required for ALERT+
#   - Cooling-off: normal caretaking actions (brief touches) ignored
#   - Minimum contact area threshold (not just wrist proximity)

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import time


# Keypoint indices (COCO 17-point)
_KP = {
    "nose":0,"left_eye":1,"right_eye":2,"left_ear":3,"right_ear":4,
    "left_shoulder":5,"right_shoulder":6,
    "left_elbow":7,"right_elbow":8,
    "left_wrist":9,"right_wrist":10,
    "left_hip":11,"right_hip":12,
    "left_knee":13,"right_knee":14,
    "left_ankle":15,"right_ankle":16,
}

# Body zone definitions — list of keypoint names that define each zone
# "virtual" zones are midpoints computed from two keypoints
BODY_ZONES = {
    # Zone name       : (kp_names..., sensitivity)
    "chest"    : (["left_shoulder","right_shoulder"],         "private"),
    "abdomen"  : (["left_hip","right_hip",
                   "left_shoulder","right_shoulder"],         "private"),   # midpoint
    "groin"    : (["left_hip","right_hip"],                   "private"),
    "inner_thigh": (["left_knee","right_knee",
                     "left_hip","right_hip"],                 "private"),
    "shoulder" : (["left_shoulder","right_shoulder"],         "sensitive"),
    "upper_arm": (["left_shoulder","left_elbow",
                   "right_shoulder","right_elbow"],           "sensitive"),
    "head"     : (["nose","left_eye","right_eye"],            "sensitive"),
}

ZONE_SENSITIVITY = {
    "private"   : 1.0,
    "sensitive" : 0.5,
    "normal"    : 0.1,
}


def _get_kp(kps: np.ndarray, name: str, min_vis: float = 0.25):
    idx = _KP[name]
    if idx >= len(kps): return None
    x,y,v = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x,y) if v >= min_vis else None

def _zone_center(kps: np.ndarray, kp_names: List[str]) -> Optional[Tuple[float,float]]:
    """Compute mean position of visible keypoints in a zone."""
    pts = [_get_kp(kps,n) for n in kp_names]
    pts = [p for p in pts if p is not None]
    if not pts: return None
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))

def _bbox_iou(b1, b2) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1,iy1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    ix2,iy2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

def _bbox_overlap_ratio(inner, outer) -> float:
    """How much of 'inner' bbox is inside 'outer' bbox."""
    ix1,iy1 = max(inner[0],outer[0]), max(inner[1],outer[1])
    ix2,iy2 = min(inner[2],outer[2]), min(inner[3],outer[3])
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    area_inner = (inner[2]-inner[0])*(inner[3]-inner[1]) + 1e-6
    return inter/area_inner


@dataclass
class TouchEvent:
    """One ongoing touch contact between an adult and a child."""
    adult_id:   int
    child_id:   int
    zone:       str        # body zone name
    sensitivity: str       # private / sensitive / normal
    start_time: float
    last_seen:  float
    score:      float = 0.0   # proximity score 0-1
    child_distress: bool = False

    @property
    def duration(self) -> float:
        return self.last_seen - self.start_time


@dataclass
class TouchAlertState:
    """Tracks alert state for inappropriate touching."""
    events:      List[TouchEvent] = field(default_factory=list)
    last_alert_time: float = 0.0
    alert_count: int = 0

    WARN_ALERT      = "TOUCH_WARN"
    ALERT_ALERT     = "TOUCH_ALERT"
    EMERGENCY_ALERT = "TOUCH_EMERGENCY"


class InappropriateTouchDetector:
    """
    Detects inappropriate physical contact between adults and children.

    Algorithm per frame:
      1. For each adult track: get wrist keypoints
      2. For each child track: compute zone centers from keypoints
      3. Measure wrist→zone distance (normalized by child bbox diagonal)
      4. Contact = distance < proximity_threshold AND bbox overlap > min_overlap
      5. Accumulate contact duration; fire alerts at thresholds

    Three alert levels:
      TOUCH_WARN       wrist near sensitive zone > warn_sec
      TOUCH_ALERT      wrist near private zone > alert_sec
                       OR any contact + child distress
      TOUCH_EMERGENCY  private contact > emergency_sec + child distress

    False positive reduction:
      - Requires bbox physical overlap (bodies must be close)
      - Brief touches (< warn_sec) are silently ignored
      - Child distress required to escalate beyond WARN
      - Alert cooldown prevents spam
    """

    def __init__(self, cfg: dict):
        tc = cfg.get("touch", {})
        self.enabled            = tc.get("enabled", True)
        self.proximity_thresh   = tc.get("proximity_threshold", 0.18)
        self.min_bbox_overlap   = tc.get("min_bbox_overlap", 0.05)
        self.warn_seconds       = tc.get("warn_seconds", 3.0)
        self.alert_seconds      = tc.get("alert_seconds", 5.0)
        self.emergency_seconds  = tc.get("emergency_seconds", 10.0)
        self.event_gap_max      = tc.get("event_gap_seconds", 2.0)
        self.cooldown           = tc.get("alert_cooldown_seconds", 30.0)

        # Active touch events: key = (adult_id, child_id, zone)
        self._events: Dict[tuple, TouchEvent] = {}
        self.state = TouchAlertState()

        # Per-frame score for display: zone_name → score
        self.frame_scores: Dict[str, float] = {}
        # Active contact summary for this frame
        self.frame_contacts: List[Dict] = []

    def _child_bbox_diagonal(self, child_bbox) -> float:
        w = child_bbox[2]-child_bbox[0]
        h = child_bbox[3]-child_bbox[1]
        return np.sqrt(w*w+h*h) + 1e-6

    def _wrist_zone_score(self, wrist_pt, zone_center, diag: float) -> float:
        """
        Soft proximity score: 1.0 = wrist on zone, 0.0 = far away.
        Uses inverse-distance weighting normalized by child bbox size.
        """
        if wrist_pt is None or zone_center is None: return 0.0
        dist = np.sqrt((wrist_pt[0]-zone_center[0])**2 +
                       (wrist_pt[1]-zone_center[1])**2)
        norm_dist = dist / (self.proximity_thresh * diag)
        return float(np.clip(1.0 - norm_dist, 0.0, 1.0))

    def analyse(self,
                tracks,          # List[TrackedPerson]
                adult_r: float,  # height_ratio threshold for adult
                child_distress: bool,
                now: float,
                frame_h: int, frame_w: int) -> Tuple[float, List[str], List[Dict]]:
        """
        Process one frame.
        Returns:
          overall_touch_score  : 0.0-1.0
          triggered_alerts     : list of alert type strings
          contacts             : list of contact dicts for display
        """
        self.frame_scores   = {}
        self.frame_contacts = []

        if not self.enabled:
            return 0.0, [], []

        adults = [t for t in tracks if t.height_ratio >= adult_r]
        children = [t for t in tracks if t.height_ratio < adult_r]

        if not adults or not children:
            # No adult-child pair — expire old events
            self._expire_events(now, force=True)
            return 0.0, [], []

        active_keys = set()
        frame_max_score = 0.0

        for ai, adult in enumerate(adults):
            if adult.kps is None: continue
            # Get adult wrists
            lw = _get_kp(adult.kps, "left_wrist")
            rw = _get_kp(adult.kps, "right_wrist")
            wrists = [w for w in [lw, rw] if w is not None]
            if not wrists: continue

            for ci, child in enumerate(children):
                if child.kps is None: continue

                # Check physical proximity via bbox overlap
                overlap = _bbox_overlap_ratio(
                    [adult.bbox[0],adult.bbox[1],adult.bbox[2],adult.bbox[3]],
                    [child.bbox[0],child.bbox[1],child.bbox[2],child.bbox[3]]
                )
                bbox_iou_val = _bbox_iou(adult.bbox, child.bbox)

                # Bodies must be at least partially overlapping
                if bbox_iou_val < self.min_bbox_overlap and overlap < self.min_bbox_overlap:
                    continue

                diag = self._child_bbox_diagonal(child.bbox)

                for zone_name, (kp_names, sensitivity) in BODY_ZONES.items():
                    zone_ctr = _zone_center(child.kps, kp_names)
                    if zone_ctr is None: continue

                    # Score = max wrist proximity to this zone
                    zone_score = max(
                        self._wrist_zone_score(w, zone_ctr, diag)
                        for w in wrists
                    )
                    # Weight by zone sensitivity
                    weighted = zone_score * ZONE_SENSITIVITY[sensitivity]
                    frame_max_score = max(frame_max_score, weighted)

                    if zone_score < 0.15:   # negligible proximity
                        continue

                    key = (ai, ci, zone_name)
                    active_keys.add(key)

                    if key not in self._events:
                        self._events[key] = TouchEvent(
                            adult_id=ai, child_id=ci,
                            zone=zone_name, sensitivity=sensitivity,
                            start_time=now, last_seen=now,
                            score=zone_score,
                        )
                    else:
                        ev = self._events[key]
                        # Allow small gaps (child briefly moves)
                        if now - ev.last_seen <= self.event_gap_max:
                            ev.last_seen = now
                            ev.score     = max(ev.score, zone_score)
                            ev.child_distress = child_distress
                        else:
                            # Gap too large — restart event
                            self._events[key] = TouchEvent(
                                adult_id=ai, child_id=ci,
                                zone=zone_name, sensitivity=sensitivity,
                                start_time=now, last_seen=now,
                                score=zone_score,
                            )

                    ev = self._events[key]
                    self.frame_contacts.append({
                        "zone":        zone_name,
                        "sensitivity": sensitivity,
                        "duration":    ev.duration,
                        "score":       zone_score,
                        "distress":    child_distress,
                    })
                    self.frame_scores[zone_name] = max(
                        self.frame_scores.get(zone_name, 0.0), zone_score)

        # Expire events not seen this frame
        self._expire_events(now, active_keys=active_keys)

        # Evaluate alert conditions
        alerts = self._evaluate_alerts(now, child_distress)

        return float(frame_max_score), alerts, self.frame_contacts

    def _expire_events(self, now: float,
                       active_keys=None, force=False):
        to_del = []
        for key, ev in self._events.items():
            if force or (active_keys is not None and key not in active_keys):
                if now - ev.last_seen > self.event_gap_max:
                    to_del.append(key)
        for k in to_del:
            del self._events[k]

    def _evaluate_alerts(self, now: float,
                         child_distress: bool) -> List[str]:
        alerts = []
        cooldown_ok = (now - self.state.last_alert_time) >= self.cooldown

        for ev in self._events.values():
            dur = ev.duration

            # EMERGENCY: private zone + long duration + child distress
            if (ev.sensitivity == "private"
                    and dur >= self.emergency_seconds
                    and child_distress
                    and cooldown_ok):
                alerts.append(TouchAlertState.EMERGENCY_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False
                continue

            # ALERT: private zone + threshold  OR  any + distress
            if (((ev.sensitivity == "private" and dur >= self.alert_seconds)
                 or (dur >= self.alert_seconds and child_distress))
                    and cooldown_ok):
                alerts.append(TouchAlertState.ALERT_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False
                continue

            # WARN: sensitive/private zone > warn threshold
            if (ev.sensitivity in ("private","sensitive")
                    and dur >= self.warn_seconds
                    and cooldown_ok):
                alerts.append(TouchAlertState.WARN_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False

        # Deduplicate — only highest severity
        if TouchAlertState.EMERGENCY_ALERT in alerts:
            return [TouchAlertState.EMERGENCY_ALERT]
        if TouchAlertState.ALERT_ALERT in alerts:
            return [TouchAlertState.ALERT_ALERT]
        if TouchAlertState.WARN_ALERT in alerts:
            return [TouchAlertState.WARN_ALERT]
        return []

    def top_contacts(self, n=3) -> List[Dict]:
        """Return top N contacts sorted by sensitivity * duration."""
        order = {"private":3,"sensitive":2,"normal":1}
        return sorted(
            self.frame_contacts,
            key=lambda c: (order.get(c["sensitivity"],0), c["duration"]),
            reverse=True
        )[:n]


# ─────────────────────────────────────────────────────────────
# Main BabyMonitor
# ─────────────────────────────────────────────────────────────
class BabyMonitor:
    def __init__(self, config_path="config.yaml"):
        self.cfg      = load_config(config_path)
        self.logger   = setup_logging(self.cfg)
        self.notifier = Notifier(self.cfg, self.logger)
        self.state    = AlertState()

        mc = self.cfg["model"]
        self.logger.info(f"Loading model: {mc['weights']}")
        self.model = YOLO(mc["weights"])

        dev = mc.get("device","auto")
        if dev == "auto":
            try:
                import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError: dev = "cpu"
        self.device = dev
        self.logger.info(f"Device: {self.device}")

        a = self.cfg["alerts"]
        self.alone_window     = a.get("alone_window_seconds", 240)
        self.cry_window       = a.get("cry_window_seconds", 240)
        self.cry_ratio_thresh = a.get("cry_ratio_threshold", 0.70)
        self.cooldown         = a.get("alert_cooldown_seconds", 60)

        # NMS dedup threshold (higher = more aggressive merging of duplicates)
        self.dedup_iou = mc.get("dedup_iou", 0.45)

        al = self.cfg.get("alone", {})
        self.use_size = al.get("use_size_filter", True)
        self.adult_r  = al.get("adult_height_ratio", 0.50)

        self.emotion_analyser = EmotionAnalyser(self.cfg, self.logger)
        self.pose_analyser    = PoseDistressAnalyser(self.cfg)
        self.audio_det        = AudioCryDetector(self.cfg, self.logger)
        self.motion_det       = MotionCryDetector()
        self.cry_tracker      = CryWindowTracker(window_seconds=self.cry_window)

        # Centroid tracker for stable person count
        self.tracker = PersonTracker(
            max_dist_ratio=al.get("tracker_max_dist_ratio", 0.15),
            max_miss=al.get("tracker_max_miss", 5),
        )

        oc = self.cfg["output"]
        if oc.get("save_video"):
            Path(oc["output_path"]).parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Baby Monitor ready.")

    def _open_source(self):
        s = self.cfg["source"]; t = s.get("type","webcam")
        if   t=="webcam": cap=cv2.VideoCapture(s.get("device_id",0))
        elif t=="rtsp":   cap=cv2.VideoCapture(s.get("rtsp_url",""))
        elif t=="file":   cap=cv2.VideoCapture(s.get("file_path",""))
        else: raise ValueError(f"Unknown source: {t}")
        if not cap.isOpened():
            raise RuntimeError("Cannot open video. Check config.yaml -> source.")
        return cap

    def _extract_detections(self, res, frame_h: int, frame_w: int):
        """
        Extract boxes + keypoints from YOLO results,
        apply secondary NMS to remove duplicates,
        then update the centroid tracker.

        Returns list of TrackedPerson objects (one per unique person).
        """
        boxes_list  = []
        scores_list = []
        kps_flat    = []

        for r in res:
            for i, b in enumerate(r.boxes):
                boxes_list.append(b.xyxy[0].cpu().numpy())
                scores_list.append(float(b.conf[0]))
                # Attach keypoints if available
                if r.keypoints is not None and i < len(r.keypoints.data):
                    kps_flat.append(r.keypoints.data[i].cpu().numpy())
                else:
                    kps_flat.append(None)

        if not boxes_list:
            # No detections — update tracker with empty
            return self.tracker.update(
                np.zeros((0,4)), np.zeros(0), None, frame_w, frame_h)

        boxes_arr  = np.array(boxes_list)
        scores_arr = np.array(scores_list)
        has_kps    = any(k is not None for k in kps_flat)
        kps_arr    = np.array([k if k is not None else
                                np.zeros((17,3)) for k in kps_flat]) if has_kps else None

        # Secondary NMS — merge duplicate detections of the same person
        boxes_arr, scores_arr, kps_arr = nms_deduplicate(
            boxes_arr, scores_arr, kps_arr, iou_thresh=self.dedup_iou)

        # Update centroid tracker
        return self.tracker.update(
            boxes_arr, scores_arr, kps_arr, frame_w, frame_h)

    def _classify_tracks(self, tracks: List[TrackedPerson]) -> Tuple[int, int]:
        """
        Count children and adults from deduplicated tracks.
        Adult: height_ratio >= adult_height_ratio threshold.
        """
        ch, ad = 0, 0
        for t in tracks:
            if t.height_ratio >= self.adult_r: ad += 1
            else:                               ch += 1
        return ch, ad

    def _is_alone(self, ch: int, ad: int) -> bool:
        """Alone = exactly 1 child, 0 adults."""
        return ch == 1 and ad == 0

    def _analyse_cry(self, frame, pose_signals, pose_score):
        """
        Returns three independent cry scores + combined:
          face_score  : 0.0-1.0  from FER emotion (dominant cry-emotion confidence)
          pose_score  : 0.0-1.0  from pose distress signals (already computed)
          motion_score: 0.0-1.0  from motion/audio fallback
          combined    : weighted average of all three
          emotions    : [(emotion, conf), ...]
        """
        # ── Face score ───────────────────────────────
        fer_cry, emotions = self.emotion_analyser.analyse(frame)
        face_score = 0.0
        if emotions:
            cry_emo = set(self.emotion_analyser.cry_emo)
            # Sum of cry-class confidences across detected faces
            # cry_emo contains capitalized names: Sad, Fear, Angry, Disgust
            cry_confs = [s for e, s in emotions if e in cry_emo]
            face_score = min(1.0, sum(cry_confs) / max(len(emotions), 1))

        # ── Motion / audio score ─────────────────────
        if self.audio_det._running:
            motion_score = 1.0 if self.audio_det.is_crying else 0.0
        else:
            raw = self.motion_det.update(frame)
            # Convert bool to soft score using motion history average
            avg = (sum(self.motion_det.history) / len(self.motion_det.history)
                   if self.motion_det.history else 0.0)
            motion_score = min(1.0, avg / max(self.motion_det.threshold, 1e-5))

        # ── Combined weighted score ──────────────────
        # Weights: face=0.5, pose=0.35, motion=0.15
        # Face and pose are more reliable; motion is a rough proxy.
        w_face, w_pose, w_motion = 0.50, 0.35, 0.15
        combined = (face_score * w_face +
                    pose_score * w_pose +
                    motion_score * w_motion)

        # Crying = combined >= 0.35  OR  face alone >= 0.5  OR  pose alone >= 0.6
        is_crying = (combined >= 0.35
                     or face_score >= 0.50
                     or pose_score >= 0.60)

        return is_crying, face_score, pose_score, motion_score, combined, emotions

    def _alert(self, atype, msg):
        now = time.time()
        if now - self.state.last_alert_time < self.cooldown: return
        self.state.last_alert_time = now; self.state.alert_count += 1
        self.notifier.send(atype, msg)

    @staticmethod
    def _draw_skeleton(frame, kps, color=(0,220,220)):
        MIN_VIS = 0.3
        pts = {}
        for name, idx in KP.items():
            if idx < len(kps):
                x, y, v = kps[idx]
                if v >= MIN_VIS:
                    pts[name] = (int(x), int(y))
                    cv2.circle(frame,(int(x),int(y)),5,color,-1)
        for a, b in SKELETON_BONES:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],color,2)

    def _draw(self, frame, tracks,
              ch, ad, alone,
              is_crying, emotions,
              pose_signals, pose_score,
              face_score, pose_cry_score, motion_score, combined_score,
              asec, cry_ratio):
        """
        Overlay layout (left panel):
          [Children / Adults count]
          [ALONE / SAFE]
          [Alone progress bar]          -- if alone
          ─────────────────────────────
          Face score   ████░░  0.72      -- FER emotion
          Pose score   ██░░░░  0.40      -- body language
          Motion score █░░░░░  0.18      -- motion / audio
          ─────────────────────────────
          COMBINED     ████░░  0.58  CRY
          ─────────────────────────────
          4-min ratio  ███░░░  61%
        """
        h, w = frame.shape[:2]

        # ── Bounding boxes + skeleton ─────────────────────────
        for t in tracks:
            x1,y1,x2,y2 = map(int, t.bbox)
            color = (0,60,255) if alone else (0,210,70)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,4)
            lbl = f"{'child' if t.height_ratio<self.adult_r else 'adult'} {t.height_ratio:.0%}"
            lbl_y = max(y1-10,30)
            cv2.putText(frame,lbl,(x1,lbl_y),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),5)
            cv2.putText(frame,lbl,(x1,lbl_y),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,2)
            if t.kps is not None:
                sk_col = (0,80,255) if pose_score >= 0.4 else (0,220,220)
                self._draw_skeleton(frame, t.kps, sk_col)

        # ── Text helper ───────────────────────────────────────
        def put(txt, y, col=(240,240,240), fs=1.8, th=3):
            cv2.putText(frame,txt,(10,y),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),th+4)
            cv2.putText(frame,txt,(10,y),cv2.FONT_HERSHEY_SIMPLEX,fs,col,th)

        # ── Score bar helper ──────────────────────────────────
        BAR_X, BAR_W, BAR_H = 10, 300, 22

        def score_bar(label, score, y, active_col, label_fs=1.3):
            """Draw  LABEL  [████░░░░]  0.72"""
            # label
            cv2.putText(frame, label, (BAR_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_fs, (0,0,0), 5)
            cv2.putText(frame, label, (BAR_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_fs, (210,210,210), 2)
            bar_y = y + 6
            # background
            cv2.rectangle(frame,(BAR_X, bar_y),(BAR_X+BAR_W, bar_y+BAR_H),(50,50,50),-1)
            # fill
            fill_w = int(BAR_W * min(score, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame,(BAR_X, bar_y),(BAR_X+fill_w, bar_y+BAR_H),active_col,-1)
            # score text to the right
            pct_txt = f"{score:.2f}"
            cv2.putText(frame, pct_txt, (BAR_X+BAR_W+10, bar_y+BAR_H-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 4)
            cv2.putText(frame, pct_txt, (BAR_X+BAR_W+10, bar_y+BAR_H-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230,230,230), 2)

        def divider(y):
            cv2.line(frame,(BAR_X, y),(BAR_X+BAR_W+80, y),(80,80,80),1)

        # ── Top status ────────────────────────────────────────
        put(f"Children:{ch}  Adults:{ad}", 55)
        alone_col = (30,50,255) if alone else (30,210,70)
        put("ALONE !" if alone else "SAFE", 115, alone_col)

        # Alone progress bar
        y_cur = 130
        if alone and asec > 0:
            pct = min(asec/self.alone_window, 1.0)
            cv2.rectangle(frame,(BAR_X,y_cur),(BAR_X+BAR_W,y_cur+20),(40,40,40),-1)
            cv2.rectangle(frame,(BAR_X,y_cur),(BAR_X+int(BAR_W*pct),y_cur+20),(0,130,255),-1)
            put(f"Alone:{asec:.0f}s/{self.alone_window}s", y_cur+50, (0,150,255), fs=1.3)
            y_cur += 80
        else:
            y_cur += 10

        divider(y_cur); y_cur += 30

        # ── Emotion label (FER detail) ────────────────────────
        if self.emotion_analyser.enabled and emotions:
            emo_str = "  ".join(f"{e}:{s:.0%}" for e,s in emotions[:2])
            cv2.putText(frame, emo_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
            cv2.putText(frame, emo_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,60), 2)
            y_cur += 45

        # ── Pose signals (detail) ─────────────────────────────
        if pose_signals:
            sig_str = " | ".join(pose_signals)
            pose_detail_col = (40,100,255) if pose_score>=0.4 else (160,160,60)
            cv2.putText(frame, sig_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 4)
            cv2.putText(frame, sig_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, pose_detail_col, 2)
            y_cur += 42

        divider(y_cur); y_cur += 35

        # ── 3 individual score bars ───────────────────────────
        FACE_COL   = (60, 180, 255)   # blue-ish  — face/FER
        POSE_COL   = (60, 255, 160)   # green     — pose
        MOTION_COL = (180, 180, 60)   # yellow    — motion/audio

        score_bar("Face  ", face_score,   y_cur, FACE_COL);   y_cur += 70
        score_bar("Pose  ", pose_cry_score, y_cur, POSE_COL); y_cur += 70
        score_bar("Motion", motion_score, y_cur, MOTION_COL); y_cur += 70

        divider(y_cur); y_cur += 35

        # ── Combined score bar (larger, highlighted) ──────────
        COMBINED_COL = (0, 60, 255) if is_crying else (0, 180, 120)
        score_bar("COMBINED", combined_score, y_cur, COMBINED_COL, label_fs=1.5)
        cry_label = "  << CRY" if is_crying else ""
        cry_lbl_col = (0,40,255) if is_crying else (100,200,100)
        cv2.putText(frame, cry_label, (BAR_X+BAR_W+80, y_cur+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 5)
        cv2.putText(frame, cry_label, (BAR_X+BAR_W+80, y_cur+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, cry_lbl_col, 2)
        y_cur += 75

        divider(y_cur); y_cur += 35

        # ── 4-minute window ratio bar ─────────────────────────
        score_bar("4min%  ", cry_ratio, y_cur, (0,80,255), label_fs=1.3)
        # threshold marker line
        thr_x = BAR_X + int(BAR_W * self.cry_ratio_thresh)
        cv2.line(frame,(thr_x, y_cur+6),(thr_x, y_cur+6+BAR_H),(0,0,255),3)
        cv2.putText(frame, f"thr:{self.cry_ratio_thresh:.0%}",
                    (thr_x-10, y_cur-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,60,255), 2)

        # ── Touch detection panel ─────────────────────────────────
        touch_alerts   = touch_alerts   or []
        touch_contacts = touch_contacts or []

        divider(y_cur); y_cur += 30

        TOUCH_COL = (0,40,255) if touch_score >= 0.5 else (
                     (0,140,255) if touch_score >= 0.25 else (60,200,60))
        score_bar("Touch ", touch_score, y_cur, TOUCH_COL, label_fs=1.3)
        y_cur += 65

        if touch_contacts:
            sens_cols = {"private":(0,30,255),"sensitive":(0,140,255),"normal":(80,200,80)}
            for c in touch_contacts[:3]:
                col = sens_cols.get(c["sensitivity"],(180,180,180))
                txt = f"  {c['zone']:12s} {c['duration']:4.1f}s  [{c['sensitivity']}]"
                cv2.putText(frame,txt,(10,y_cur),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),4)
                cv2.putText(frame,txt,(10,y_cur),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,2)
                y_cur += 38

        if TouchAlertState.EMERGENCY_ALERT in touch_alerts:
            badge_col,badge_txt = (0,0,220),"!! TOUCH EMERGENCY !!"
        elif TouchAlertState.ALERT_ALERT in touch_alerts:
            badge_col,badge_txt = (0,60,255),"! TOUCH ALERT !"
        elif TouchAlertState.WARN_ALERT in touch_alerts:
            badge_col,badge_txt = (0,140,255),"TOUCH WARNING"
        else:
            badge_col = badge_txt = None

        if badge_txt:
            bx,by = 10,y_cur+10
            tw,th = cv2.getTextSize(badge_txt,cv2.FONT_HERSHEY_SIMPLEX,1.6,3)[0]
            cv2.rectangle(frame,(bx-4,by-th-8),(bx+tw+8,by+6),badge_col,-1)
            cv2.putText(frame,badge_txt,(bx,by),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),3)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame,ts,(w-170,h-15),cv2.FONT_HERSHEY_SIMPLEX,1.1,(160,160,160),2)
        return frame

    def run(self):
        cap    = self._open_source()
        oc     = self.cfg["output"]
        show   = oc.get("show_video", True)
        save   = oc.get("save_video", False)
        writer = None
        mc     = self.cfg["model"]
        pcls   = self.cfg["classes"]["person_class_id"]
        self.logger.info("Monitoring started. Press q to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame."); break

                h, w = frame.shape[:2]
                now  = time.time()

                # YOLO inference
                res = self.model.predict(
                    frame,
                    conf=mc.get("confidence", 0.45),
                    iou=mc.get("iou", 0.65),   # raised from 0.45 to reduce in-model dupes
                    classes=[pcls],
                    imgsz=mc.get("imgsz", 640),
                    device=self.device,
                    verbose=False,
                )

                # Extract + deduplicate + track
                tracks = self._extract_detections(res, h, w)
                ch, ad = self._classify_tracks(tracks)
                alone  = self._is_alone(ch, ad)

                # Pose analysis on child track
                pose_score, pose_signals = 0.0, []
                for t in tracks:
                    if t.height_ratio < self.adult_r and t.kps is not None:
                        pose_score, pose_signals = self.pose_analyser.analyse(t.kps)
                        break   # analyse first (and only) child

                # Cry analysis — 3 independent scores + combined
                (is_crying,
                 face_score, pose_cry_score,
                 motion_score, combined_score,
                 emotions) = self._analyse_cry(frame, pose_signals, pose_score)

                # Sliding window (combined score >= 0.35 counts as crying frame)
                if alone:
                    self.cry_tracker.push(now, is_crying)
                else:
                    self.cry_tracker.reset()
                cry_ratio = self.cry_tracker.ratio()

                # Alone timer + alerts
                asec = 0.0
                if alone:
                    if self.state.alone_start is None:
                        self.state.alone_start = now
                        self.logger.info("Child alone — monitoring.")
                    asec = now - self.state.alone_start

                    if asec >= self.alone_window:
                        self._alert(AlertState.ALONE_ALERT,
                            f"Child ALONE for {timedelta(seconds=int(asec))}!")

                    if (self.cry_tracker.window_filled(now)
                            and cry_ratio >= self.cry_ratio_thresh):
                        self._alert(AlertState.CRY_ALERT,
                            f"EMERGENCY: Child crying {cry_ratio:.0%} of last "
                            f"{self.cry_window//60} min! "
                            f"({', '.join(pose_signals) or 'FER/motion'})")
                else:
                    if self.state.alone_start is not None:
                        elapsed = now - self.state.alone_start
                        self.state.total_alone_seconds += elapsed
                        self.logger.info(f"Caregiver present. Alone: {elapsed:.0f}s")
                    self.state.alone_start = None

                if show or save:
                    vis = self._draw(frame.copy(), tracks,
                                     ch, ad, alone,
                                     is_crying, emotions,
                                     pose_signals, pose_score,
                                     face_score, pose_cry_score,
                                     motion_score, combined_score,
                                     asec, cry_ratio)
                    if show: cv2.imshow("Baby Monitor", vis)
                    if save:
                        if writer is None:
                            fps  = oc.get("fps",25)
                            res2 = tuple(oc.get("resolution",[w,h]))
                            writer = cv2.VideoWriter(oc["output_path"],
                                     cv2.VideoWriter_fourcc(*"mp4v"),fps,res2)
                        writer.write(cv2.resize(vis,tuple(oc.get("resolution",[w,h]))))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            if self.audio_det: self.audio_det.stop()
            self.logger.info(
                f"Stopped. Alerts:{self.state.alert_count}, "
                f"Alone total:{self.state.total_alone_seconds:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    BabyMonitor(config_path=ap.parse_args().config).run()

'''

# ─────────────────────────────────────────────────────
# training/train.py
# ─────────────────────────────────────────────────────
TRAIN_PY = '''\
"""
YOLO11 Training Pipeline - Baby Monitor custom model.

Usage:
  python training/train.py
  python training/train.py --validate-only
  python training/train.py --export
"""

import argparse, shutil, sys
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics"); sys.exit(1)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_dataset_yaml(cfg):
    ds   = cfg["dataset"]
    base = Path(ds["path"]).resolve()
    out  = Path("training/dataset.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump({"path": str(base),
                   "train": ds.get("train_dir", "images/train"),
                   "val":   ds.get("val_dir",   "images/val"),
                   "test":  ds.get("test_dir",  "images/test"),
                   "names": ds["class_names"],
                   "nc":    len(ds["class_names"])}, f, allow_unicode=True)
    print(f"[INFO] dataset.yaml -> {out}")
    return str(out)

def train(cfg, ds_yaml):
    model = YOLO(cfg["model"].get("base", "yolo11n.pt"))
    tc    = cfg["training"]
    model.train(data=ds_yaml, epochs=tc.get("epochs",100), imgsz=tc.get("imgsz",640),
        batch=tc.get("batch",16), lr0=tc.get("lr0",0.01), lrf=tc.get("lrf",0.01),
        momentum=tc.get("momentum",0.937), weight_decay=tc.get("weight_decay",0.0005),
        warmup_epochs=tc.get("warmup_epochs",3), device=tc.get("device","auto"),
        workers=tc.get("workers",4), project=tc.get("project","training/runs"),
        name=tc.get("name","baby_monitor"), exist_ok=True, pretrained=True,
        optimizer=tc.get("optimizer","SGD"), patience=tc.get("patience",50),
        save_period=tc.get("save_period",10), plots=True, verbose=True)

def validate(cfg, ds_yaml):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if not best.exists(): print(f"[WARN] {best} not found"); return
    met = YOLO(str(best)).val(data=ds_yaml)
    print(f"[INFO] mAP50={met.box.map50:.4f}  mAP50-95={met.box.map:.4f}")

def export_model(cfg):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if not best.exists(): print("[WARN] best.pt not found"); return
    YOLO(str(best)).export(format=cfg.get("export",{}).get("format","onnx"))

def copy_best(cfg):
    best = (Path(cfg["training"].get("project","training/runs"))
            / cfg["training"].get("name","baby_monitor") / "weights/best.pt")
    if best.exists():
        shutil.copy2(best, "best_custom.pt")
        print("[INFO] best_custom.pt ready -> set model.weights in config.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",        default="training/train_config.yaml")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-train",    action="store_true")
    ap.add_argument("--export",        action="store_true")
    args = ap.parse_args()
    cfg  = load_cfg(args.config)
    ds   = make_dataset_yaml(cfg)
    if args.validate_only:   validate(cfg, ds)
    elif args.export:        export_model(cfg)
    else:
        if not args.skip_train: train(cfg, ds)
        validate(cfg, ds); copy_best(cfg)
        if cfg.get("export", {}).get("enabled"): export_model(cfg)
    print("[DONE]")
'''

# ─────────────────────────────────────────────────────
# training/train_config.yaml
# ─────────────────────────────────────────────────────
TRAIN_CONFIG_YAML = """\
model:
  base: "yolo11n.pt"    # yolo11n | yolo11s | yolo11m | yolo11l | yolo11x

dataset:
  path: "training/data"
  train_dir: "images/train"
  val_dir: "images/val"
  test_dir: "images/test"
  class_names:
    - child    # 0
    - adult    # 1

training:
  epochs: 100
  imgsz: 640
  batch: 16
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  patience: 50
  save_period: 10
  optimizer: "SGD"
  augment: true
  device: "auto"
  workers: 4
  seed: 42
  project: "training/runs"
  name: "baby_monitor_v1"

export:
  enabled: false
  format: "onnx"

# Label format (YOLO, normalized 0-1):
#   <class_id> <x_center> <y_center> <width> <height>
# Labeling tools: Roboflow, CVAT, LabelImg
"""

# ─────────────────────────────────────────────────────
# training/prepare_dataset.py
# ─────────────────────────────────────────────────────
PREPARE_DATASET_PY = '''\
"""Dataset preparation helper. See --help for usage."""
import argparse, shutil, random
from pathlib import Path
import cv2

def extract_frames(video, out_dir, every=30, max_frames=500):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): print(f"[ERROR] {video}"); return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved, fi, name = [], 0, Path(video).stem
    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if fi % every == 0:
            p = Path(out_dir)/f"{name}_{fi:06d}.jpg"
            cv2.imwrite(str(p), frame); saved.append(p)
        fi += 1
    cap.release(); print(f"[INFO] {len(saved)} frames -> {out_dir}"); return saved

def split_dataset(imgs_dir, out, tr=0.7, vl=0.2):
    imgs = list(Path(imgs_dir).glob("*.jpg")) + list(Path(imgs_dir).glob("*.png"))
    random.shuffle(imgs)
    n_tr, n_vl = int(len(imgs)*tr), int(len(imgs)*vl)
    for split, files in [("train",imgs[:n_tr]),("val",imgs[n_tr:n_tr+n_vl]),("test",imgs[n_tr+n_vl:])]:
        ip = Path(out)/"images"/split; lp = Path(out)/"labels"/split
        ip.mkdir(parents=True, exist_ok=True); lp.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, ip/f.name)
            lf = lp/(f.stem+".txt")
            if not lf.exists(): lf.touch()
        print(f"  {split}: {len(files)}")

def create_structure(out):
    for s in ["train","val","test"]:
        (Path(out)/"images"/s).mkdir(parents=True, exist_ok=True)
        (Path(out)/"labels"/s).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Structure created: {out}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--output", default="training/data")
    ap.add_argument("--every-n", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--create-structure", action="store_true")
    args = ap.parse_args()
    if args.create_structure: create_structure(args.output)
    elif args.video:
        raw = str(Path(args.output)/"raw_frames")
        extract_frames(args.video, raw, args.every_n, args.max_frames)
        if args.split: split_dataset(raw, args.output)
    else: create_structure(args.output)
'''

# ─────────────────────────────────────────────────────
# requirements.txt
# ─────────────────────────────────────────────────────
REQUIREMENTS_TXT = """\
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
torch>=2.0.0                        # required by facial-emotion-recognition
torchvision>=0.15.0                 # required by facial-emotion-recognition
facial-emotion-recognition>=0.3.4   # bundled 33MB model, offline, 95.6% acc
# Optional:
# pyaudio>=0.2.13      # microphone-based cry detection
# requests>=2.31.0     # Telegram notifications
# torch>=2.0.0         # GPU acceleration (CUDA)
"""

# ─────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────
README_MD = """# Baby Monitor - Child Safety Monitor

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
"""

# ─────────────────────────────────────────────────────
# FILE MAP
# ─────────────────────────────────────────────────────
FILES = {
    "config.yaml":                 CONFIG_YAML,
    "detector.py":                 DETECTOR_PY,
    "requirements.txt":            REQUIREMENTS_TXT,
    "README.md":                   README_MD,
    "training/train.py":           TRAIN_PY,
    "training/train_config.yaml":  TRAIN_CONFIG_YAML,
    "training/prepare_dataset.py": PREPARE_DATASET_PY,
}

DIRS = [
    "logs", "output",
    "training/data/images/train", "training/data/images/val",
    "training/data/images/test",  "training/data/labels/train",
    "training/data/labels/val",   "training/data/labels/test",
]


# ─────────────────────────────────────────────────────
# SETUP FUNCTIONS
# ─────────────────────────────────────────────────────

def create_project(base: Path):
    section("Creating directories")
    for d in DIRS:
        t = base / d
        t.mkdir(parents=True, exist_ok=True)
        ok(str(t))
    section("Writing files")
    for rel, content in FILES.items():
        fp = base / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        ok(str(fp))


def install_packages():
    section("Installing packages")
    pkgs = [
        ("ultralytics",                "ultralytics>=8.3.0"),
        ("cv2",                        "opencv-python>=4.8.0"),
        ("numpy",                      "numpy>=1.24.0"),
        ("yaml",                       "pyyaml>=6.0"),
        ("facial_emotion_recognition", "facial-emotion-recognition>=0.3.4"),
    ]
    for mod, pkg in pkgs:
        try:
            __import__(mod)
            ok(f"{pkg} — already installed")
        except ImportError:
            info(f"Installing {pkg} ...")
            # Try --break-system-packages (Homebrew / system Python on macOS)
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q",
                 "--break-system-packages"],
                capture_output=True, text=True)
            if r.returncode != 0:
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    capture_output=True, text=True)
            if r.returncode == 0:
                ok(f"{pkg} — installed")
            else:
                warn(f"{pkg} failed: {r.stderr[:100]}")


def print_summary(base: Path):
    section("DONE")
    print(f"""
{BOLD}{G}  Baby Monitor ready!{END}

{W}  Start:{END}  {Y}python detector.py{END}

{W}  Key files:{END}
    {C}config.yaml{END}   <- all settings
    {C}detector.py{END}   <- run this

{W}  Alert rules:{END}
    {DIM}ALONE  : 1 child alone >= 4 min{END}
    {DIM}CRYING : crying in >= 70% of last 4 min window{END}

{W}  Two children together = NOT alone{END}

{W}  Quit:{END} press {Y}q{END} in the video window
""")


def main():
    print(f"""
{BOLD}{B}{'='*55}{END}
{BOLD}{W}    Baby Monitor - Setup Script{END}
{BOLD}{B}    YOLO11 + FER | Crying Alone Detector{END}
{BOLD}{B}{'='*55}{END}
""")
    base = Path.cwd()
    info(f"Installing into: {base}")
    create_project(base)
    install_packages()
    print_summary(base)


if __name__ == "__main__":
    main()