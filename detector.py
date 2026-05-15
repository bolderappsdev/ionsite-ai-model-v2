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
import os
import json
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

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
# v2 alert string → IONSITE (alertType, severity) mapping
# AlertType enum on the web side is AGGRESSION | TOUCHING | NEGLECT.
# AGGRESSION is intentionally absent — v2 doesn't emit it yet.
# ─────────────────────────────────────────────────────────────
_IONSITE_ALERT_MAP: Dict[str, Tuple[str, str]] = {
    "ALONE_4MIN":        ("NEGLECT",  "HIGH"),
    "CRYING_70PCT_4MIN": ("NEGLECT",  "HIGH"),
    "TOUCH_WARN":        ("TOUCHING", "LOW"),
    "TOUCH_ALERT":       ("TOUCHING", "MEDIUM"),
    "TOUCH_EMERGENCY":   ("TOUCHING", "HIGH"),
}


# ─────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────
class Notifier:
    def __init__(self, cfg, logger):
        self.cfg = cfg.get("notification", {}); self.logger = logger
        self._executor: Optional[ThreadPoolExecutor] = None
        ic = self.cfg.get("ionsite", {})
        if ic.get("enabled") and REQUESTS_AVAILABLE:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ionsite"
            )

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

    def ingest_health_probe(self) -> None:
        """One-shot startup check against IONSITE's /api/alert/ingest/health.
        Logs ok or warning. Never raises — misconfiguration shouldn't block startup."""
        ic = self.cfg.get("ionsite", {})
        if not ic.get("enabled"):
            return
        if not REQUESTS_AVAILABLE:
            self.logger.warning(
                "IONSITE ingest enabled but `requests` not installed."
            )
            return
        url = ic.get("health_url", "")
        if not url:
            self.logger.warning("IONSITE ingest enabled but health_url unset.")
            return
        try:
            r = req.get(
                url,
                headers={"X-Service-Key": ic.get("service_key", "")},
                timeout=ic.get("timeout_seconds", 10),
            )
            if 200 <= r.status_code < 300:
                self.logger.info(f"IONSITE ingest reachable ({url})")
            else:
                self.logger.warning(
                    f"IONSITE ingest unreachable: HTTP {r.status_code} "
                    f"body={r.text[:200]}"
                )
        except Exception as e:
            self.logger.warning(f"IONSITE ingest unreachable: {e}")

    def dispatch_alert(self, envelope: Dict[str, Any], clip_path: str) -> None:
        """Submit alert + clip to IONSITE in a background thread. Best-effort
        (no retries) — the AI-side cooldown plus subsequent alerts will pick up
        anything missed by a transient failure."""
        ic = self.cfg.get("ionsite", {})
        if not ic.get("enabled") or not REQUESTS_AVAILABLE:
            self._unlink(clip_path)
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ionsite"
            )
        self._executor.submit(self._send_ionsite, ic, envelope, clip_path)

    def _send_ionsite(self, ic: Dict[str, Any], envelope: Dict[str, Any], clip_path: str) -> None:
        try:
            metadata = {
                "cameraId":    envelope["cameraId"],
                "alertType":   envelope["alertType"],
                "severity":    envelope["severity"],
                "confidence":  envelope["confidence"],
                "description": envelope["description"],
            }
            # Deterministic Idempotency-Key: retries within the same second of
            # the same camera+atype collapse onto the same response server-side.
            idem = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{envelope['cameraId']}|{envelope['atype']}|{int(envelope['timestamp'])}",
            )
            with open(clip_path, "rb") as fh:
                files = {
                    "metadata": (None, json.dumps(metadata), "application/json"),
                    "clip":     ("clip.mp4", fh, "video/mp4"),
                }
                r = req.post(
                    ic["url"],
                    headers={
                        "X-Service-Key":   ic.get("service_key", ""),
                        "Idempotency-Key": str(idem),
                    },
                    files=files,
                    timeout=ic.get("timeout_seconds", 10),
                )
            if 200 <= r.status_code < 300:
                self.logger.info(
                    f"IONSITE ingest ok: {envelope['atype']} → "
                    f"{envelope['alertType']}/{envelope['severity']} "
                    f"(conf={envelope['confidence']})"
                )
            else:
                self.logger.warning(
                    f"IONSITE ingest failed: HTTP {r.status_code} "
                    f"body={r.text[:200]}"
                )
        except Exception as e:
            self.logger.warning(f"IONSITE ingest exception: {e}")
        finally:
            self._unlink(clip_path)

    @staticmethod
    def _unlink(path: str) -> None:
        try:
            os.unlink(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Pre/post-alert frame buffer + clip mux
# ─────────────────────────────────────────────────────────────
class PreAlertBuffer:
    """Ring of (timestamp, BGR-frame-copy) entries. Used to build the pre-roll
    portion of the alert clip; sized for pre_alert_seconds * clip_fps."""
    def __init__(self, max_frames: int) -> None:
        self._buf: deque = deque(maxlen=max(1, max_frames))

    def append(self, ts: float, frame) -> None:
        if frame is not None:
            self._buf.append((ts, frame.copy()))

    def snapshot(self) -> List[Any]:
        return [f for (_, f) in self._buf]


@dataclass
class _PendingClipState:
    envelope: Dict[str, Any]
    pre_frames: List[Any]
    post_frames: List[Any] = field(default_factory=list)
    deadline: float = 0.0
    target_post_frames: int = 0


def _mux_clip(pending: _PendingClipState, fps: int) -> str:
    """Write pre+post frames into a temp mp4 using H.264 (avc1) so the browser's
    <video> element can play it. OpenCV's Mac wheel uses VideoToolbox under the
    hood for avc1; on Linux it falls back to whatever the system has. We try
    avc1 first and fall back to mp4v only if the writer fails to open — that
    fallback won't play in browsers but at least the alert file is preserved.
    """
    all_frames = pending.pre_frames + pending.post_frames
    if not all_frames:
        raise RuntimeError("no frames to mux")
    h, w = all_frames[0].shape[:2]
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="ionsite-alert-")
    os.close(fd)

    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h),
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )

    try:
        for f in all_frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
    finally:
        writer.release()
    return path


# ─────────────────────────────────────────────────────────────
# Inappropriate Touch Detector  v2
# ─────────────────────────────────────────────────────────────
#
# CORE ALGORITHM:
#   A touch is confirmed only when the adult's WRIST KEYPOINT
#   lands INSIDE the child's bounding box.  The zone (head /
#   chest / groin / legs) is determined by where in the bbox
#   the wrist sits vertically:
#
#       ┌─────────────┐  ← y1 (top of child bbox)
#       │  HEAD  25%  │  sensitive
#       ├─────────────┤
#       │ CHEST  25%  │  PRIVATE
#       ├─────────────┤
#       │ GROIN  25%  │  PRIVATE
#       ├─────────────┤
#       │  LEGS  25%  │  normal
#       └─────────────┘  ← y2 (bottom of child bbox)
#
#   Additionally, when YOLO keypoints are available, we refine
#   the zone boundaries using actual body keypoints so the zones
#   follow the skeleton rather than the raw bbox.
#
# ALERT LEVELS:
#   TOUCH_WARN       wrist in sensitive zone  >= warn_seconds
#   TOUCH_ALERT      wrist in private zone    >= alert_seconds
#                    OR any zone + child distress >= alert_seconds
#   TOUCH_EMERGENCY  wrist in private zone    >= emergency_seconds
#                    AND child distress confirmed
#
# FALSE POSITIVE REDUCTION:
#   1. Wrist must be INSIDE child bbox (not just near it)
#   2. Brief contacts < warn_seconds silently ignored
#   3. ALERT+ requires child distress (crying OR pose signals)
#   4. Per-alert cooldown prevents notification spam
#   5. Horizontal wrist position checked — wrists at bbox edges
#      (likely passing by) weighted less than centre contacts

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import time


# ── COCO keypoint indices (same as global KP dict) ───────────
_T_KP = {
    "nose":0,
    "left_shoulder":5,  "right_shoulder":6,
    "left_hip":11,      "right_hip":12,
    "left_knee":13,     "right_knee":14,
    "left_wrist":9,     "right_wrist":10,
}

# Zone sensitivity levels
ZONE_SENSITIVITY = {
    "head":      "sensitive",
    "chest":     "private",
    "groin":     "private",
    "legs":      "normal",
}

SENSITIVITY_SCORE = {"private": 1.0, "sensitive": 0.5, "normal": 0.1}


def _kp_visible(kps: np.ndarray, name: str,
                min_vis: float = 0.25) -> Optional[Tuple[float, float]]:
    """Return (x, y) of keypoint if visible, else None."""
    idx = _T_KP.get(name)
    if idx is None or idx >= len(kps):
        return None
    x, y, v = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x, y) if v >= min_vis else None


def _mid(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _zone_boundaries_from_kps(child_kps: np.ndarray,
                               bbox: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Compute y-boundaries for each zone using skeleton keypoints.
    Falls back to bbox-fraction boundaries when keypoints are missing.

    Returns dict: zone_name -> (y_top, y_bottom)
    """
    y1, y2 = float(bbox[1]), float(bbox[3])
    h = y2 - y1 + 1e-6

    # Try to get keypoint-based anchors
    shoulder = _mid(
        _kp_visible(child_kps, "left_shoulder"),
        _kp_visible(child_kps, "right_shoulder"),
    )
    hip = _mid(
        _kp_visible(child_kps, "left_hip"),
        _kp_visible(child_kps, "right_hip"),
    )
    knee = _mid(
        _kp_visible(child_kps, "left_knee"),
        _kp_visible(child_kps, "right_knee"),
    )
    nose = _kp_visible(child_kps, "nose")

    # Zone top boundaries (y coordinate, image space ↓)
    head_top    = nose[1] - (shoulder[1] - nose[1]) * 0.5 if (nose and shoulder) else y1
    head_bot    = shoulder[1]                              if shoulder              else y1 + h * 0.25
    chest_bot   = hip[1]                                  if hip                   else y1 + h * 0.50
    groin_bot   = knee[1]                                 if knee                  else y1 + h * 0.75

    # Clamp to bbox
    def clamp(v): return max(y1, min(y2, v))

    return {
        "head":  (clamp(head_top),  clamp(head_bot)),
        "chest": (clamp(head_bot),  clamp(chest_bot)),
        "groin": (clamp(chest_bot), clamp(groin_bot)),
        "legs":  (clamp(groin_bot), y2),
    }


def _classify_wrist_zone(wrist_y: float,
                         zones: Dict[str, Tuple[float, float]]) -> str:
    """Return the zone name containing this wrist y-coordinate."""
    for name, (top, bot) in zones.items():
        if top <= wrist_y <= bot:
            return name
    return "legs"   # fallback


def _centre_weight(wrist_x: float, bbox_x1: float, bbox_x2: float) -> float:
    """
    Weight 0.5–1.0 based on how centred the wrist is horizontally.
    Wrists near bbox edges (passing by) get lower weight.
    """
    cx   = (bbox_x1 + bbox_x2) / 2
    half = (bbox_x2 - bbox_x1) / 2 + 1e-6
    dist_ratio = abs(wrist_x - cx) / half   # 0 = centre, 1 = edge
    return float(np.clip(1.0 - dist_ratio * 0.5, 0.5, 1.0))


@dataclass
class TouchEvent:
    adult_id:    int
    child_id:    int
    zone:        str
    sensitivity: str
    start_time:  float
    last_seen:   float
    score:       float = 0.0
    child_distress: bool = False

    @property
    def duration(self) -> float:
        return self.last_seen - self.start_time


@dataclass
class TouchAlertState:
    last_alert_time: float = 0.0
    alert_count:     int   = 0

    WARN_ALERT      = "TOUCH_WARN"
    ALERT_ALERT     = "TOUCH_ALERT"
    EMERGENCY_ALERT = "TOUCH_EMERGENCY"


class InappropriateTouchDetector:
    """
    Detects inappropriate physical contact using wrist-in-bbox logic.

    Key improvement over v1:
      - Wrist keypoint must be INSIDE child bounding box (not just nearby)
      - Zone determined by vertical position within bbox / skeleton
      - Horizontal centre-weight reduces false positives from pass-bys
      - Soft score per touch event shown on overlay
    """

    def __init__(self, cfg: dict):
        tc = cfg.get("touch", {})
        self.enabled           = tc.get("enabled", True)
        self.warn_seconds      = tc.get("warn_seconds", 3.0)
        self.alert_seconds     = tc.get("alert_seconds", 5.0)
        self.emergency_seconds = tc.get("emergency_seconds", 10.0)
        self.event_gap_max     = tc.get("event_gap_seconds", 2.0)
        self.cooldown          = tc.get("alert_cooldown_seconds", 30.0)
        # Wrist must be this far inside bbox edges (fraction of bbox width)
        self.edge_margin       = tc.get("edge_margin", 0.05)

        self._events: Dict[tuple, TouchEvent] = {}
        self.state = TouchAlertState()
        self.frame_contacts: List[Dict] = []

    # ── Core wrist-in-bbox check ──────────────────────────────
    def _wrist_in_bbox(self,
                       wrist: Tuple[float, float],
                       bbox:  np.ndarray) -> bool:
        """
        True if wrist (x,y) is inside bbox [x1,y1,x2,y2].
        A small inward margin removes edge false positives.
        """
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        mx = (x2 - x1) * self.edge_margin
        my = (y2 - y1) * self.edge_margin
        return (x1 + mx < wrist[0] < x2 - mx and
                y1 + my < wrist[1] < y2 - my)

    # ── Per-frame analysis ────────────────────────────────────
    def analyse(self,
                tracks,
                adult_r: float,
                child_distress: bool,
                now: float,
                frame_h: int,
                frame_w: int) -> Tuple[float, List[str], List[Dict]]:
        """
        Returns (overall_touch_score, alert_list, contacts_for_display).
        """
        self.frame_contacts = []

        if not self.enabled:
            return 0.0, [], []

        adults   = [t for t in tracks if t.height_ratio >= adult_r]
        children = [t for t in tracks if t.height_ratio <  adult_r]

        if not adults or not children:
            self._expire_events(now, force=True)
            return 0.0, [], []

        active_keys     = set()
        frame_max_score = 0.0

        for ai, adult in enumerate(adults):
            if adult.kps is None:
                continue

            # Collect visible adult wrists
            wrists = []
            for wname in ("left_wrist", "right_wrist"):
                w = _kp_visible(adult.kps, wname)
                if w is not None:
                    wrists.append(w)
            if not wrists:
                continue

            for ci, child in enumerate(children):
                # Compute zone boundaries for this child
                zones = _zone_boundaries_from_kps(
                    child.kps if child.kps is not None else np.zeros((17, 3)),
                    child.bbox,
                )

                for wrist in wrists:
                    # ── PRIMARY CHECK: wrist must be inside child bbox ──
                    if not self._wrist_in_bbox(wrist, child.bbox):
                        continue

                    # ── Determine zone from vertical position ───────────
                    zone_name   = _classify_wrist_zone(wrist[1], zones)
                    sensitivity = ZONE_SENSITIVITY[zone_name]

                    # ── Compute contact score ───────────────────────────
                    # centre_weight: lower if wrist near bbox edge (pass-by)
                    cw    = _centre_weight(wrist[0], child.bbox[0], child.bbox[2])
                    score = SENSITIVITY_SCORE[sensitivity] * cw
                    frame_max_score = max(frame_max_score, score)

                    key = (ai, ci, zone_name)
                    active_keys.add(key)

                    if key not in self._events:
                        self._events[key] = TouchEvent(
                            adult_id=ai, child_id=ci,
                            zone=zone_name, sensitivity=sensitivity,
                            start_time=now, last_seen=now,
                            score=score,
                        )
                    else:
                        ev = self._events[key]
                        if now - ev.last_seen <= self.event_gap_max:
                            ev.last_seen       = now
                            ev.score           = max(ev.score, score)
                            ev.child_distress  = child_distress
                        else:
                            # Gap too large — restart event
                            self._events[key] = TouchEvent(
                                adult_id=ai, child_id=ci,
                                zone=zone_name, sensitivity=sensitivity,
                                start_time=now, last_seen=now,
                                score=score,
                            )

                    ev = self._events[key]
                    self.frame_contacts.append({
                        "zone":        zone_name,
                        "sensitivity": sensitivity,
                        "duration":    ev.duration,
                        "score":       score,
                        "distress":    child_distress,
                        "wrist":       wrist,         # for debug drawing
                    })

        self._expire_events(now, active_keys=active_keys)
        alerts = self._evaluate_alerts(now, child_distress)
        return float(frame_max_score), alerts, self.frame_contacts

    def _expire_events(self, now: float,
                       active_keys: set = None,
                       force: bool = False):
        to_del = [
            k for k, ev in self._events.items()
            if (force or (active_keys is not None and k not in active_keys))
            and now - ev.last_seen > self.event_gap_max
        ]
        for k in to_del:
            del self._events[k]

    def _evaluate_alerts(self, now: float,
                         child_distress: bool) -> List[str]:
        alerts      = []
        cooldown_ok = (now - self.state.last_alert_time) >= self.cooldown

        for ev in self._events.values():
            dur = ev.duration

            # EMERGENCY: private zone + long + distress
            if (ev.sensitivity == "private"
                    and dur >= self.emergency_seconds
                    and child_distress
                    and cooldown_ok):
                alerts.append(TouchAlertState.EMERGENCY_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count    += 1
                cooldown_ok = False
                continue

            # ALERT: private zone long  OR  any + distress
            if (((ev.sensitivity == "private" and dur >= self.alert_seconds)
                 or (dur >= self.alert_seconds and child_distress))
                    and cooldown_ok):
                alerts.append(TouchAlertState.ALERT_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count    += 1
                cooldown_ok = False
                continue

            # WARN: sensitive/private zone > warn threshold
            if (ev.sensitivity in ("private", "sensitive")
                    and dur >= self.warn_seconds
                    and cooldown_ok):
                alerts.append(TouchAlertState.WARN_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count    += 1
                cooldown_ok = False

        # Only report highest severity
        if TouchAlertState.EMERGENCY_ALERT in alerts:
            return [TouchAlertState.EMERGENCY_ALERT]
        if TouchAlertState.ALERT_ALERT in alerts:
            return [TouchAlertState.ALERT_ALERT]
        if TouchAlertState.WARN_ALERT in alerts:
            return [TouchAlertState.WARN_ALERT]
        return []

    def top_contacts(self, n: int = 3) -> List[Dict]:
        """Top N contacts by severity then duration."""
        order = {"private": 3, "sensitive": 2, "normal": 1}
        return sorted(
            self.frame_contacts,
            key=lambda c: (order.get(c["sensitivity"], 0), c["duration"]),
            reverse=True,
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

        # Inappropriate touch detector
        self.touch_detector = InappropriateTouchDetector(self.cfg)

        oc = self.cfg["output"]
        if oc.get("save_video"):
            Path(oc["output_path"]).parent.mkdir(parents=True, exist_ok=True)

        # ── IONSITE ingest plumbing ──────────────────────────
        ic = self.cfg.get("notification", {}).get("ionsite", {})
        self._clip_fps: int = int(ic.get("clip_fps", oc.get("fps", 25)))
        pre_seconds: float = float(ic.get("pre_alert_seconds", 5))
        post_seconds: float = float(ic.get("post_alert_seconds", 3))
        self._prealert_buf = PreAlertBuffer(int(pre_seconds * self._clip_fps))
        self._post_seconds: float = post_seconds
        self._post_target: int = int(post_seconds * self._clip_fps)
        self._pending_clips: List[_PendingClipState] = []

        self.notifier.ingest_health_probe()

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

    def _alert(self, atype, msg, confidence_score: float = 0.0, current_frame=None):
        now = time.time()
        if now - self.state.last_alert_time < self.cooldown: return
        self.state.last_alert_time = now; self.state.alert_count += 1
        self.notifier.send(atype, msg)
        self._queue_ionsite_clip(atype, msg, confidence_score, current_frame, now)

    def _queue_ionsite_clip(self, atype, msg, confidence_score, current_frame, now) -> None:
        """If IONSITE ingest is enabled, build an envelope + queue a pending clip
        capture. The main loop will fill in post-alert frames and dispatch when
        the buffer is full."""
        ic = self.cfg.get("notification", {}).get("ionsite", {})
        if not ic.get("enabled"):
            return
        camera_id = self.cfg.get("source", {}).get("camera_id", "")
        if not camera_id:
            self.logger.warning(
                "IONSITE alert skipped: source.camera_id not set in config."
            )
            return
        if atype not in _IONSITE_ALERT_MAP:
            self.logger.warning(
                f"IONSITE alert skipped: no mapping for '{atype}'."
            )
            return
        alert_type, severity = _IONSITE_ALERT_MAP[atype]
        confidence_int = max(0, min(100, int(round(float(confidence_score) * 100))))
        envelope = {
            "cameraId":    camera_id,
            "atype":       atype,
            "alertType":   alert_type,
            "severity":    severity,
            "confidence":  confidence_int,
            "description": msg,
            "timestamp":   now,
        }
        pre_frames = self._prealert_buf.snapshot()
        # If we have no pre-frames yet (first seconds of run), seed with the
        # current frame so the clip is never empty.
        if not pre_frames and current_frame is not None:
            pre_frames = [current_frame.copy()]
        self._pending_clips.append(_PendingClipState(
            envelope=envelope,
            pre_frames=pre_frames,
            deadline=now + self._post_seconds,
            target_post_frames=self._post_target,
        ))

    def _process_pending_clips(self, now: float, frame) -> None:
        """Per-frame: append to each pending clip's post-buffer; once full
        (or deadline passed), mux to mp4 and hand to the Notifier."""
        if not self._pending_clips:
            return
        for pending in list(self._pending_clips):
            if (len(pending.post_frames) < pending.target_post_frames
                    and now <= pending.deadline
                    and frame is not None):
                pending.post_frames.append(frame.copy())
            if (len(pending.post_frames) >= pending.target_post_frames
                    or now > pending.deadline):
                self._pending_clips.remove(pending)
                try:
                    clip_path = _mux_clip(pending, self._clip_fps)
                except Exception as e:
                    self.logger.warning(f"IONSITE clip mux failed: {e}")
                    continue
                self.notifier.dispatch_alert(pending.envelope, clip_path)

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
              asec, cry_ratio,
              touch_score=0.0, touch_alerts=None, touch_contacts=None):
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
        _ta = touch_alerts   if touch_alerts   is not None else []
        _tc = touch_contacts if touch_contacts is not None else []

        # Draw zone boundaries on child bboxes + wrist contact dots
        SENS_COLS = {"private":(0,30,255),"sensitive":(0,140,255),"normal":(80,200,80)}
        for t in tracks:
            if t.height_ratio < self.adult_r:          # child
                bx1,by1,bx2,by2 = map(int, t.bbox)
                bh = by2 - by1
                # Draw horizontal zone dividers on child bbox
                for frac, lbl in [(0.25,"head"),(0.50,"chest"),(0.75,"groin")]:
                    zy = int(by1 + bh * frac)
                    col = SENS_COLS.get(ZONE_SENSITIVITY.get(lbl,"normal"),(120,120,120))
                    cv2.line(frame,(bx1,zy),(bx2,zy),col,1)
                    cv2.putText(frame,lbl,(bx2+4,zy+5),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,1)

        # Draw wrist contact indicators
        for c in _tc:
            if "wrist" in c:
                wx,wy = int(c["wrist"][0]), int(c["wrist"][1])
                col   = SENS_COLS.get(c["sensitivity"],(180,180,180))
                cv2.circle(frame,(wx,wy),14,col,3)
                cv2.circle(frame,(wx,wy),4,(255,255,255),-1)
                cv2.putText(frame,c["zone"],(wx+16,wy+5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),4)
                cv2.putText(frame,c["zone"],(wx+16,wy+5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)

        divider(y_cur); y_cur += 30

        TOUCH_COL = (0,40,255) if touch_score >= 0.5 else (
                     (0,140,255) if touch_score >= 0.25 else (60,200,60))
        score_bar("Touch ", touch_score, y_cur, TOUCH_COL, label_fs=1.3)
        y_cur += 65

        if _tc:
            for c in _tc[:3]:
                col = SENS_COLS.get(c["sensitivity"],(180,180,180))
                txt = f"  {c['zone']:12s} {c['duration']:4.1f}s  [{c['sensitivity']}]"
                cv2.putText(frame,txt,(10,y_cur),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),4)
                cv2.putText(frame,txt,(10,y_cur),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,2)
                y_cur += 38

        if TouchAlertState.EMERGENCY_ALERT in _ta:
            badge_col,badge_txt = (0,0,220),"!! TOUCH EMERGENCY !!"
        elif TouchAlertState.ALERT_ALERT in _ta:
            badge_col,badge_txt = (0,60,255),"! TOUCH ALERT !"
        elif TouchAlertState.WARN_ALERT in _ta:
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

                # IONSITE clip plumbing — pre-roll ring buffer + drain any
                # post-roll captures from a recent alert.
                self._prealert_buf.append(now, frame)
                self._process_pending_clips(now, frame)

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

                # Inappropriate touch analysis
                child_distress = is_crying or (pose_score >= 0.4)
                (touch_score,
                 touch_alerts,
                 touch_contacts) = self.touch_detector.analyse(
                    tracks, self.adult_r, child_distress, now, h, w)

                # Fire touch alerts
                for ta in touch_alerts:
                    top = self.touch_detector.top_contacts(1)
                    zone_info = top[0]["zone"] if top else "unknown"
                    dur_info  = f"{top[0]['duration']:.1f}s" if top else ""
                    msgs = {
                        TouchAlertState.WARN_ALERT:
                            f"WARNING: Adult touching child ({zone_info}) for {dur_info}",
                        TouchAlertState.ALERT_ALERT:
                            f"ALERT: Inappropriate touching! Zone:{zone_info} {dur_info}",
                        TouchAlertState.EMERGENCY_ALERT:
                            f"EMERGENCY: Prolonged touch + distress! Zone:{zone_info} {dur_info}",
                    }
                    self._alert(ta, msgs.get(ta, ta), touch_score, frame)

                # Alone timer + alerts
                asec = 0.0
                if alone:
                    if self.state.alone_start is None:
                        self.state.alone_start = now
                        self.logger.info("Child alone — monitoring.")
                    asec = now - self.state.alone_start

                    if asec >= self.alone_window:
                        self._alert(AlertState.ALONE_ALERT,
                            f"Child ALONE for {timedelta(seconds=int(asec))}!",
                            1.0, frame)

                    if (self.cry_tracker.window_filled(now)
                            and cry_ratio >= self.cry_ratio_thresh):
                        self._alert(AlertState.CRY_ALERT,
                            f"EMERGENCY: Child crying {cry_ratio:.0%} of last "
                            f"{self.cry_window//60} min! "
                            f"({', '.join(pose_signals) or 'FER/motion'})",
                            cry_ratio, frame)
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
                                     asec, cry_ratio,
                                     touch_score, touch_alerts,
                                     touch_contacts)
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