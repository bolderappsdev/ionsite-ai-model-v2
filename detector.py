"""
Baby Monitor - Child Distress & Safety Detection
YOLO11-pose + Facial Emotion Recognition + Audio FFT + Inappropriate Touch.

═══════════════════════════════════════════════════════════════
DISTRESS DETECTION LOGIC:
═══════════════════════════════════════════════════════════════
Tizim bolaning DISTRESS holatlarini multi-modal aniqlaydi:

  CRYING (yig'lash):
    face  → Sad emotion dominant
    audio → 300-1000 Hz cry frekvensiyasi
    pose  → head_droop, torso_curl

  FEAR (qo'rqish):
    face  → Fear / Surprise emotion
    pose  → arms_raised (himoya), body_shake

  ANGER / TANTRUM (g'azab / isteriya):
    face  → Angry emotion
    pose  → body_shake, arms_raised
    audio → baland tovush

  PAIN / DISGUST:
    face  → Disgust / Angry
    pose  → torso_curl, lying_down

UMUMIY DISTRESS = is_crying  (any kind of negative state).
4-min window davomida shu holat 70%+ ushlansa, EMERGENCY alert.

═══════════════════════════════════════════════════════════════
CRYING DETECTION BUG FIXES:
═══════════════════════════════════════════════════════════════
  1. face_score: TO'G'RI softmax sum (eskida `/len(emotions)` BUG edi)
  2. FER endi CHILD BBOX ichida ishlaydi (butun frame emas)
  3. Haarcascade alt2 + bola yuzi uchun tuned params
  4. Yuz topilmaganda fallback — child bbox top-40% (frame center emas)
  5. CryWindowTracker faqat FRESH measurementlarni saqlaydi
  6. is_crying multi-signal voting (2+ kuchli yoki bittasi juda kuchli)
  7. Motion weight kamaytirildi
  8. Audio TO'LIQ combined logikaga qo'shildi (alohida emas)

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
# Deduplication NMS
# ─────────────────────────────────────────────────────────────
def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms_deduplicate(boxes_xyxy, scores, kps_list, iou_thresh=0.45):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, kps_list
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        suppress = []
        for j in rest:
            if iou(boxes_xyxy[i], boxes_xyxy[j]) >= iou_thresh:
                suppress.append(j)
        suppress_set = set(suppress)
        order = np.array([x for x in rest if x not in suppress_set])
    kept_boxes = boxes_xyxy[keep]
    kept_scores = scores[keep]
    kept_kps = kps_list[keep] if kps_list is not None else None
    return kept_boxes, kept_scores, kept_kps


# ─────────────────────────────────────────────────────────────
# Centroid tracker
# ─────────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    bbox: np.ndarray
    centroid: np.ndarray
    height_ratio: float
    kps: Optional[np.ndarray] = None
    miss_count: int = 0


class PersonTracker:
    def __init__(self, max_dist_ratio=0.15, max_miss=5):
        self.tracks: List[TrackedPerson] = []
        self.max_dist_ratio = max_dist_ratio
        self.max_miss = max_miss

    def update(self, boxes, scores, kps_list, frame_w, frame_h):
        diag = np.sqrt(frame_w**2 + frame_h**2) + 1e-5
        max_dist = self.max_dist_ratio * diag

        new_centroids = np.array(
            [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
        ) if len(boxes) > 0 else np.zeros((0, 2))

        matched_new = set()
        matched_track = set()

        assignments: Dict[int, int] = {}
        for ti, track in enumerate(self.tracks):
            best_dist, best_di = 1e9, -1
            for di in range(len(new_centroids)):
                if di in matched_new: continue
                d = np.linalg.norm(new_centroids[di] - track.centroid)
                if d < best_dist:
                    best_dist, best_di = d, di
            if best_di >= 0 and best_dist < max_dist:
                assignments[ti] = best_di
                matched_new.add(best_di)
                matched_track.add(ti)

        for ti, di in assignments.items():
            t = self.tracks[ti]
            t.bbox = boxes[di]
            t.centroid = new_centroids[di]
            t.height_ratio = (boxes[di][3] - boxes[di][1]) / frame_h
            t.kps = kps_list[di] if kps_list is not None else None
            t.miss_count = 0

        for ti in range(len(self.tracks)):
            if ti not in matched_track:
                self.tracks[ti].miss_count += 1

        for di in range(len(boxes)):
            if di not in matched_new:
                self.tracks.append(TrackedPerson(
                    bbox=boxes[di],
                    centroid=new_centroids[di],
                    height_ratio=(boxes[di][3] - boxes[di][1]) / frame_h,
                    kps=kps_list[di] if kps_list is not None else None,
                ))

        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]
        return self.tracks


# ─────────────────────────────────────────────────────────────
# Sliding-window cry tracker  (FIX: fresh-only push)
# ─────────────────────────────────────────────────────────────
class CryWindowTracker:
    def __init__(self, window_seconds=240, min_push_interval=0.2):
        self.window = window_seconds
        self._buf: deque = deque()
        self._last_pushed_ts = 0.0
        self._min_interval = min_push_interval

    def push(self, ts, is_crying, is_fresh=True):
        if not is_fresh: return
        if ts - self._last_pushed_ts < self._min_interval: return
        self._buf.append((ts, is_crying))
        self._last_pushed_ts = ts
        cutoff = ts - self.window
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def ratio(self):
        if not self._buf: return 0.0
        return sum(1 for _, c in self._buf if c) / len(self._buf)

    def window_filled(self, now):
        if not self._buf: return False
        return (now - self._buf[0][0]) >= self.window

    def reset(self):
        self._buf.clear()
        self._last_pushed_ts = 0.0


# ─────────────────────────────────────────────────────────────
# Alert state
# ─────────────────────────────────────────────────────────────
@dataclass
class AlertState:
    alone_start: Optional[float] = None
    last_alert_time: float = 0.0
    alert_count: int = 0
    total_alone_seconds: float = 0.0

    ALONE_ALERT = "ALONE_4MIN"
    CRY_ALERT = "CRYING_70PCT_4MIN"


# ─────────────────────────────────────────────────────────────
# Pose distress analyser
# ─────────────────────────────────────────────────────────────
class PoseDistressAnalyser:
    """
    5 ta body-language distress signali:
      arms_raised — qo'l yelka ustida (qo'rqish, himoya, isteriya)
      torso_curl  — bosh yelka pastida (yig'lash, gavda burishishi)
      head_droop  — bosh osilgan (yig'lash, charchoq, og'riq)
      lying_down  — gavda gorizontal (yiqilgan / yotgan)
      body_shake  — tez tebranish (isteriya, dahshat, og'ir yig'lash)
    """
    MIN_VIS = 0.3

    def __init__(self, cfg):
        pc = cfg.get("pose", {})
        self.enabled = pc.get("enabled", True)
        self.score_threshold = pc.get("distress_score_threshold", 0.4)
        self._pos_history: deque = deque(maxlen=8)

    @staticmethod
    def _get(kps, name):
        idx = KP[name]
        if idx >= len(kps): return None
        x, y, v = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
        return (x, y, v) if v >= PoseDistressAnalyser.MIN_VIS else None

    @staticmethod
    def _mid(a, b):
        if a is None or b is None: return None
        return ((a[0]+b[0])/2, (a[1]+b[1])/2)

    def _arms_raised(self, kps):
        ls, rs = self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder")
        lw, rw = self._get(kps,"left_wrist"), self._get(kps,"right_wrist")
        raised, total = 0, 0
        for sh, wr in [(ls,lw),(rs,rw)]:
            if sh and wr:
                total += 1
                if wr[1] < sh[1]: raised += 1
        return 1.0 if (total > 0 and raised == total) else 0.0

    def _torso_curl(self, kps):
        nose = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder"))
        hip_mid = self._mid(self._get(kps,"left_hip"), self._get(kps,"right_hip"))
        if None in (nose, sh_mid, hip_mid): return 0.0
        body_h = abs(hip_mid[1] - sh_mid[1]) + 1e-5
        drop = nose[1] - sh_mid[1]
        return 1.0 if drop > 0.5 * body_h else 0.0

    def _head_droop(self, kps):
        nose = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder"))
        if None in (nose, sh_mid): return 0.0
        return 1.0 if nose[1] >= sh_mid[1] else 0.0

    def _lying_down(self, kps):
        sh_mid = self._mid(self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder"))
        hip_mid = self._mid(self._get(kps,"left_hip"), self._get(kps,"right_hip"))
        if None in (sh_mid, hip_mid): return 0.0
        dx = abs(hip_mid[0]-sh_mid[0])
        dy = abs(hip_mid[1]-sh_mid[1]) + 1e-5
        return 1.0 if dx/dy > 2.0 else 0.0

    def _body_shake(self, kps):
        nose = self._get(kps,"nose")
        sh_mid = self._mid(self._get(kps,"left_shoulder"), self._get(kps,"right_shoulder"))
        ref = nose or sh_mid
        if ref is None: return 0.0
        self._pos_history.append((ref[0], ref[1]))
        if len(self._pos_history) < 8: return 0.0
        spread = np.std([p[0] for p in self._pos_history]) + \
                 np.std([p[1] for p in self._pos_history])
        return 1.0 if spread > 8.0 else 0.0

    def analyse(self, kps):
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


# ═════════════════════════════════════════════════════════════
# EmotionAnalyser — FIXED
# ═════════════════════════════════════════════════════════════
try:
    import torch
    import torchvision.transforms as tv_transforms
    from facial_emotion_recognition.networks import NetworkV2
    _FER_PKG_AVAILABLE = True
except ImportError:
    _FER_PKG_AVAILABLE = False


class EmotionAnalyser:
    """
    7-class emotion recognition (Sad/Fear/Angry/Disgust/Happy/Surprise/Neutral).

    DISTRESS emotionlar = cry_emotions config (default: Sad, Fear, Angry, Disgust).
    Bu yerda "cry" = umumiy distress (yig'lash + qo'rqish + g'azab + jirkanish).
    """

    EMOTIONS = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}

    def __init__(self, cfg, logger):
        ec = cfg.get("emotion", {})
        self.enabled = ec.get("enabled", True) and _FER_PKG_AVAILABLE
        self.cry_emo = set(ec.get("cry_emotions", ["Sad","Fear","Angry","Disgust"]))
        self.min_conf = ec.get("min_confidence", 0.40)
        self.every_n = max(1, int(ec.get("run_every_n_frames", 2)))
        self._fc = 0
        self._last_cry = False
        self._last_emo: List[Tuple[str,float]] = []
        self._last_cry_score = 0.0
        self._last_dominant = "Neutral"

        if not self.enabled:
            if not _FER_PKG_AVAILABLE:
                logger.warning("facial-emotion-recognition not installed. "
                               "Run: pip install facial-emotion-recognition")
            self._net = None
            return

        import os
        model_path = os.path.join(
            os.path.dirname(__import__("facial_emotion_recognition").__file__),
            "model", "model.pkl")

        self._device = torch.device("cpu")
        self._net = NetworkV2(in_c=1, nl=32, out_f=7).to(self._device)
        state = torch.load(model_path, map_location="cpu")
        self._net.load_state_dict(state["network"])
        self._net.eval()
        logger.info(f"Emotion model loaded (acc={state.get('accuracy',0):.2%})")

        self._transform = tv_transforms.Compose([
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((48, 48)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        import cv2 as _cv2
        cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._face_det = _cv2.CascadeClassifier(cascade_path)
        if self._face_det.empty():
            cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_det = _cv2.CascadeClassifier(cascade_path)
        logger.info(f"Face detector: {Path(cascade_path).name}")

    def _predict_face(self, gray_face):
        if gray_face.size == 0: return []
        tensor = self._transform(gray_face).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._net(tensor)
            scores = torch.softmax(out, dim=1).squeeze().tolist()
        return sorted(
            [(self.EMOTIONS[i], scores[i]) for i in range(7)],
            key=lambda x: x[1], reverse=True)

    def analyse(self, frame, child_bbox=None):
        """
        Returns (is_distress, top_emotions, distress_score, dominant_emotion, is_fresh).
          is_distress  : bool
          top_emotions : top-3 [(name, conf), ...]
          distress_score : 0..1 — distress emotion softmax yig'indisi (TO'G'RI)
          dominant_emotion : eng kuchli emotion nomi (Sad/Fear/Angry/Disgust/...)
          is_fresh : bool
        """
        if not self.enabled or self._net is None:
            return False, [], 0.0, "Disabled", False

        self._fc += 1
        if self._fc % self.every_n != 0:
            return (self._last_cry, self._last_emo,
                    self._last_cry_score, self._last_dominant, False)

        # FIX: child bbox ichida qidirish
        if child_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in child_bbox]
            h_f, w_f = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w_f, x2 + pad_x); y2 = min(h_f, y2 + pad_y)
            roi_color = frame[y1:y2, x1:x2]
            if roi_color.size == 0:
                self._last_cry = False; self._last_emo = []
                self._last_cry_score = 0.0; self._last_dominant = "Neutral"
                return False, [], 0.0, "Neutral", True
            gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._face_det.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        all_preds: List[List[Tuple[str,float]]] = []

        if len(faces) == 0:
            h_roi, w_roi = gray.shape
            if h_roi > 30 and w_roi > 30:
                head_crop = gray[0:int(h_roi * 0.4), :]
                if head_crop.size > 0:
                    preds = self._predict_face(head_crop)
                    if preds: all_preds.append(preds)
        else:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]
            for (x, y, fw, fh) in faces:
                crop = gray[y:y+fh, x:x+fw]
                preds = self._predict_face(crop)
                if preds: all_preds.append(preds)

        if not all_preds:
            self._last_cry = False; self._last_emo = []
            self._last_cry_score = 0.0; self._last_dominant = "Neutral"
            return False, [], 0.0, "Neutral", True

        # TO'G'RI distress score = cry emotion softmax sum
        cry_scores_per_face = []
        for preds in all_preds:
            cry_sum = sum(score for emo, score in preds if emo in self.cry_emo)
            cry_scores_per_face.append(cry_sum)
        cry_score = max(cry_scores_per_face)

        # Dominant emotion (eng kuchli yuzdan)
        top_emo, top_sc = all_preds[0][0]
        dominant = top_emo

        is_cry = False
        if top_emo in self.cry_emo and top_sc >= self.min_conf:
            is_cry = True
        if cry_score >= 0.55:
            is_cry = True

        display_emotions = all_preds[0][:3]

        self._last_cry = is_cry
        self._last_emo = display_emotions
        self._last_cry_score = cry_score
        self._last_dominant = dominant
        return is_cry, display_emotions, cry_score, dominant, True


# ─────────────────────────────────────────────────────────────
# Motion fallback
# ─────────────────────────────────────────────────────────────
class MotionCryDetector:
    def __init__(self, threshold=0.15, history=10):
        self.threshold = threshold
        self.history = deque(maxlen=history)
        self.prev_frame = None

    def update(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(21,21),0)
        if self.prev_frame is None:
            self.prev_frame = gray; return False
        delta = cv2.absdiff(self.prev_frame, gray)
        _, th = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        score = th.sum()/(th.shape[0]*th.shape[1]*255)
        self.prev_frame = gray
        self.history.append(score)
        return (sum(self.history)/len(self.history)) > self.threshold


# ─────────────────────────────────────────────────────────────
# Audio FFT cry detector — TO'LIQ INTEGRATSIYA QILINGAN
# ─────────────────────────────────────────────────────────────
class AudioCryDetector:
    """
    FFT asosida mikrofondan cry frekvensiyasini aniqlaydi.
    - is_crying  : bool (binary)
    - cry_score  : float 0..1 (cry frekvensiyalari energiyasi nisbati)
    - rms_db     : volume in dB
    Endi cry_score umumiy combined logikaga TO'LIQ qo'shiladi.
    """
    def __init__(self, cfg, logger):
        ac = cfg.get("audio", {})
        self.is_crying = False
        self.cry_score = 0.0           # soft score 0..1
        self.rms_db = -80.0
        self._running = False
        self.logger = logger

        if not (ac.get("enabled", False) and AUDIO_AVAILABLE):
            if ac.get("enabled", False) and not AUDIO_AVAILABLE:
                logger.warning("pyaudio not installed — audio disabled.")
            return

        self._cfg = ac
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()
        logger.info("Audio FFT cry detector started.")

    def _listen(self):
        import pyaudio as pa_mod
        pa = pa_mod.PyAudio()
        rate = self._cfg.get("sample_rate", 16000)
        chunk = self._cfg.get("chunk_size", 1024)
        f_min = self._cfg.get("cry_frequency_min", 300)
        f_max = self._cfg.get("cry_frequency_max", 1000)
        amp_t = self._cfg.get("cry_amplitude_threshold", 0.3)
        try:
            stream = pa.open(format=pa_mod.paFloat32, channels=1,
                             rate=rate, input=True, frames_per_buffer=chunk)
            while self._running:
                data = np.frombuffer(
                    stream.read(chunk, exception_on_overflow=False),
                    dtype=np.float32)

                # RMS volume (dB)
                rms = float(np.sqrt(np.mean(data ** 2)))
                self.rms_db = float(20 * np.log10(rms + 1e-7))

                max_amp = float(np.max(np.abs(data)))

                if max_amp > amp_t:
                    fft = np.abs(np.fft.rfft(data))
                    freqs = np.fft.rfftfreq(len(data), 1.0/rate)
                    mask = (freqs >= f_min) & (freqs <= f_max)
                    cry_band = fft[mask].sum()
                    total = fft.sum() + 1e-9
                    # SOFT score: cry frekvensiyalari ulushini volume bilan vaznli
                    ratio = cry_band / total           # 0..1
                    volume_factor = min(1.0, max_amp / max(amp_t, 1e-5))
                    self.cry_score = float(np.clip(ratio * volume_factor, 0.0, 1.0))
                    self.is_crying = ratio > 0.30
                else:
                    self.cry_score *= 0.7   # gradual decay
                    self.is_crying = False
        except Exception as e:
            self.logger.error(f"Audio thread error: {e}")
        finally:
            pa.terminate()

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
                         json={"chat_id":tg["chat_id"],"text":msg}, timeout=5)
            except Exception: pass
        em = self.cfg.get("email", {})
        if em.get("enabled"):
            try:
                import smtplib; from email.mime.text import MIMEText
                m = MIMEText(msg); m["Subject"] = f"[BabyMonitor] {atype}"
                m["From"] = em["sender"]; m["To"] = em["recipient"]
                with smtplib.SMTP(em["smtp_server"], em["smtp_port"]) as s:
                    s.starttls(); s.login(em["sender"], em["password"]); s.send_message(m)
            except Exception: pass


# ─────────────────────────────────────────────────────────────
# Inappropriate Touch Detector
# ─────────────────────────────────────────────────────────────
_T_KP = {
    "nose":0,
    "left_shoulder":5, "right_shoulder":6,
    "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14,
    "left_wrist":9, "right_wrist":10,
}

ZONE_SENSITIVITY = {
    "head":  "sensitive",
    "chest": "private",
    "groin": "private",
    "legs":  "normal",
}

SENSITIVITY_SCORE = {"private": 1.0, "sensitive": 0.5, "normal": 0.1}


def _kp_visible(kps, name, min_vis=0.25):
    idx = _T_KP.get(name)
    if idx is None or idx >= len(kps): return None
    x, y, v = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x, y) if v >= min_vis else None


def _mid(a, b):
    if a is None or b is None: return None
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)


def _zone_boundaries_from_kps(child_kps, bbox):
    y1, y2 = float(bbox[1]), float(bbox[3])
    h = y2 - y1 + 1e-6

    shoulder = _mid(_kp_visible(child_kps, "left_shoulder"),
                    _kp_visible(child_kps, "right_shoulder"))
    hip = _mid(_kp_visible(child_kps, "left_hip"),
               _kp_visible(child_kps, "right_hip"))
    knee = _mid(_kp_visible(child_kps, "left_knee"),
                _kp_visible(child_kps, "right_knee"))
    nose = _kp_visible(child_kps, "nose")

    head_top  = nose[1] - (shoulder[1] - nose[1]) * 0.5 if (nose and shoulder) else y1
    head_bot  = shoulder[1] if shoulder else y1 + h * 0.25
    chest_bot = hip[1] if hip else y1 + h * 0.50
    groin_bot = knee[1] if knee else y1 + h * 0.75

    def clamp(v): return max(y1, min(y2, v))

    return {
        "head":  (clamp(head_top),  clamp(head_bot)),
        "chest": (clamp(head_bot),  clamp(chest_bot)),
        "groin": (clamp(chest_bot), clamp(groin_bot)),
        "legs":  (clamp(groin_bot), y2),
    }


def _classify_wrist_zone(wrist_y, zones):
    for name, (top, bot) in zones.items():
        if top <= wrist_y <= bot: return name
    return "legs"


def _centre_weight(wrist_x, bbox_x1, bbox_x2):
    cx = (bbox_x1 + bbox_x2) / 2
    half = (bbox_x2 - bbox_x1) / 2 + 1e-6
    dist_ratio = abs(wrist_x - cx) / half
    return float(np.clip(1.0 - dist_ratio * 0.5, 0.5, 1.0))


@dataclass
class TouchEvent:
    adult_id: int
    child_id: int
    zone: str
    sensitivity: str
    start_time: float
    last_seen: float
    score: float = 0.0
    child_distress: bool = False

    @property
    def duration(self): return self.last_seen - self.start_time


@dataclass
class TouchAlertState:
    last_alert_time: float = 0.0
    alert_count: int = 0

    WARN_ALERT = "TOUCH_WARN"
    ALERT_ALERT = "TOUCH_ALERT"
    EMERGENCY_ALERT = "TOUCH_EMERGENCY"


class InappropriateTouchDetector:
    def __init__(self, cfg):
        tc = cfg.get("touch", {})
        self.enabled = tc.get("enabled", True)
        self.warn_seconds = tc.get("warn_seconds", 3.0)
        self.alert_seconds = tc.get("alert_seconds", 5.0)
        self.emergency_seconds = tc.get("emergency_seconds", 10.0)
        self.event_gap_max = tc.get("event_gap_seconds", 2.0)
        self.cooldown = tc.get("alert_cooldown_seconds", 30.0)
        self.edge_margin = tc.get("edge_margin", 0.05)

        self._events: Dict[tuple, TouchEvent] = {}
        self.state = TouchAlertState()
        self.frame_contacts: List[Dict] = []

    def _wrist_in_bbox(self, wrist, bbox):
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        mx = (x2 - x1) * self.edge_margin
        my = (y2 - y1) * self.edge_margin
        return (x1 + mx < wrist[0] < x2 - mx and
                y1 + my < wrist[1] < y2 - my)

    def analyse(self, tracks, adult_r, child_distress, now, frame_h, frame_w):
        self.frame_contacts = []
        if not self.enabled: return 0.0, [], []

        adults = [t for t in tracks if t.height_ratio >= adult_r]
        children = [t for t in tracks if t.height_ratio < adult_r]

        if not adults or not children:
            self._expire_events(now, force=True)
            return 0.0, [], []

        active_keys = set()
        frame_max_score = 0.0

        for ai, adult in enumerate(adults):
            if adult.kps is None: continue
            wrists = []
            for wname in ("left_wrist", "right_wrist"):
                w = _kp_visible(adult.kps, wname)
                if w is not None: wrists.append(w)
            if not wrists: continue

            for ci, child in enumerate(children):
                zones = _zone_boundaries_from_kps(
                    child.kps if child.kps is not None else np.zeros((17, 3)),
                    child.bbox)

                for wrist in wrists:
                    if not self._wrist_in_bbox(wrist, child.bbox): continue
                    zone_name = _classify_wrist_zone(wrist[1], zones)
                    sensitivity = ZONE_SENSITIVITY[zone_name]
                    cw = _centre_weight(wrist[0], child.bbox[0], child.bbox[2])
                    score = SENSITIVITY_SCORE[sensitivity] * cw
                    frame_max_score = max(frame_max_score, score)

                    key = (ai, ci, zone_name)
                    active_keys.add(key)

                    if key not in self._events:
                        self._events[key] = TouchEvent(
                            adult_id=ai, child_id=ci,
                            zone=zone_name, sensitivity=sensitivity,
                            start_time=now, last_seen=now, score=score)
                    else:
                        ev = self._events[key]
                        if now - ev.last_seen <= self.event_gap_max:
                            ev.last_seen = now
                            ev.score = max(ev.score, score)
                            ev.child_distress = child_distress
                        else:
                            self._events[key] = TouchEvent(
                                adult_id=ai, child_id=ci,
                                zone=zone_name, sensitivity=sensitivity,
                                start_time=now, last_seen=now, score=score)

                    ev = self._events[key]
                    self.frame_contacts.append({
                        "zone": zone_name, "sensitivity": sensitivity,
                        "duration": ev.duration, "score": score,
                        "distress": child_distress, "wrist": wrist,
                    })

        self._expire_events(now, active_keys=active_keys)
        alerts = self._evaluate_alerts(now, child_distress)
        return float(frame_max_score), alerts, self.frame_contacts

    def _expire_events(self, now, active_keys=None, force=False):
        to_del = [
            k for k, ev in self._events.items()
            if (force or (active_keys is not None and k not in active_keys))
            and now - ev.last_seen > self.event_gap_max
        ]
        for k in to_del: del self._events[k]

    def _evaluate_alerts(self, now, child_distress):
        alerts = []
        cooldown_ok = (now - self.state.last_alert_time) >= self.cooldown

        for ev in self._events.values():
            dur = ev.duration

            if (ev.sensitivity == "private" and dur >= self.emergency_seconds
                    and child_distress and cooldown_ok):
                alerts.append(TouchAlertState.EMERGENCY_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False
                continue

            if (((ev.sensitivity == "private" and dur >= self.alert_seconds)
                 or (dur >= self.alert_seconds and child_distress))
                    and cooldown_ok):
                alerts.append(TouchAlertState.ALERT_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False
                continue

            if (ev.sensitivity in ("private", "sensitive")
                    and dur >= self.warn_seconds and cooldown_ok):
                alerts.append(TouchAlertState.WARN_ALERT)
                self.state.last_alert_time = now
                self.state.alert_count += 1
                cooldown_ok = False

        if TouchAlertState.EMERGENCY_ALERT in alerts: return [TouchAlertState.EMERGENCY_ALERT]
        if TouchAlertState.ALERT_ALERT in alerts: return [TouchAlertState.ALERT_ALERT]
        if TouchAlertState.WARN_ALERT in alerts: return [TouchAlertState.WARN_ALERT]
        return []

    def top_contacts(self, n=3):
        order = {"private": 3, "sensitive": 2, "normal": 1}
        return sorted(self.frame_contacts,
            key=lambda c: (order.get(c["sensitivity"], 0), c["duration"]),
            reverse=True)[:n]


# ═════════════════════════════════════════════════════════════
# Main BabyMonitor
# ═════════════════════════════════════════════════════════════
class BabyMonitor:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logging(self.cfg)
        self.notifier = Notifier(self.cfg, self.logger)
        self.state = AlertState()

        mc = self.cfg["model"]
        self.logger.info(f"Loading model: {mc['weights']}")
        self.model = YOLO(mc["weights"])

        dev = mc.get("device", "auto")
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError: dev = "cpu"
        self.device = dev
        self.logger.info(f"Device: {self.device}")

        a = self.cfg["alerts"]
        self.alone_window = a.get("alone_window_seconds", 240)
        self.cry_window = a.get("cry_window_seconds", 240)
        self.cry_ratio_thresh = a.get("cry_ratio_threshold", 0.70)
        self.cooldown = a.get("alert_cooldown_seconds", 60)

        self.dedup_iou = mc.get("dedup_iou", 0.45)

        al = self.cfg.get("alone", {})
        self.use_size = al.get("use_size_filter", True)
        self.adult_r = al.get("adult_height_ratio", 0.50)

        # Cry fusion weights (config orqali)
        fc = self.cfg.get("fusion", {})
        self.w_face = float(fc.get("face_weight", 0.40))
        self.w_audio = float(fc.get("audio_weight", 0.35))
        self.w_pose = float(fc.get("pose_weight", 0.20))
        self.w_motion = float(fc.get("motion_weight", 0.05))
        self.combined_thresh = float(fc.get("combined_threshold", 0.45))
        self.strong_signal_thresh = float(fc.get("strong_signal_threshold", 0.55))
        self.very_strong_thresh = float(fc.get("very_strong_threshold", 0.70))

        self.emotion_analyser = EmotionAnalyser(self.cfg, self.logger)
        self.pose_analyser = PoseDistressAnalyser(self.cfg)
        self.audio_det = AudioCryDetector(self.cfg, self.logger)
        self.motion_det = MotionCryDetector()
        self.cry_tracker = CryWindowTracker(window_seconds=self.cry_window)

        self.tracker = PersonTracker(
            max_dist_ratio=al.get("tracker_max_dist_ratio", 0.15),
            max_miss=al.get("tracker_max_miss", 5))

        self.touch_detector = InappropriateTouchDetector(self.cfg)

        oc = self.cfg["output"]
        if oc.get("save_video"):
            Path(oc["output_path"]).parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Baby Monitor ready.")

    def _open_source(self):
        s = self.cfg["source"]; t = s.get("type", "webcam")
        if   t == "webcam": cap = cv2.VideoCapture(s.get("device_id", 0))
        elif t == "rtsp":   cap = cv2.VideoCapture(s.get("rtsp_url", ""))
        elif t == "file":   cap = cv2.VideoCapture(s.get("file_path", ""))
        else: raise ValueError(f"Unknown source: {t}")
        if not cap.isOpened():
            raise RuntimeError("Cannot open video. Check config.yaml -> source.")
        return cap

    def _extract_detections(self, res, frame_h, frame_w):
        boxes_list, scores_list, kps_flat = [], [], []
        for r in res:
            for i, b in enumerate(r.boxes):
                boxes_list.append(b.xyxy[0].cpu().numpy())
                scores_list.append(float(b.conf[0]))
                if r.keypoints is not None and i < len(r.keypoints.data):
                    kps_flat.append(r.keypoints.data[i].cpu().numpy())
                else:
                    kps_flat.append(None)

        if not boxes_list:
            return self.tracker.update(
                np.zeros((0,4)), np.zeros(0), None, frame_w, frame_h)

        boxes_arr = np.array(boxes_list)
        scores_arr = np.array(scores_list)
        has_kps = any(k is not None for k in kps_flat)
        kps_arr = np.array([k if k is not None else np.zeros((17,3))
                            for k in kps_flat]) if has_kps else None

        boxes_arr, scores_arr, kps_arr = nms_deduplicate(
            boxes_arr, scores_arr, kps_arr, iou_thresh=self.dedup_iou)

        return self.tracker.update(
            boxes_arr, scores_arr, kps_arr, frame_w, frame_h)

    def _classify_tracks(self, tracks):
        ch, ad = 0, 0
        for t in tracks:
            if t.height_ratio >= self.adult_r: ad += 1
            else: ch += 1
        return ch, ad

    def _is_alone(self, ch, ad):
        return ch == 1 and ad == 0

    def _pick_child_bbox(self, tracks):
        largest_area = 0.0
        bbox = None
        for t in tracks:
            if t.height_ratio < self.adult_r:
                x1, y1, x2, y2 = t.bbox
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    bbox = t.bbox
        return bbox

    # ═════════════════════════════════════════════════════════
    # MULTI-MODAL DISTRESS ANALYSIS  (face + audio + pose + motion)
    # ═════════════════════════════════════════════════════════
    def _analyse_cry(self, frame, pose_signals, pose_score,
                     child_bbox: Optional[np.ndarray] = None):
        """
        TO'LIQ multi-modal distress analysis.

        Bola DISTRESS holatlari:
          - Yig'lash   (Sad + audio + head_droop/torso_curl)
          - Qo'rqish   (Fear + arms_raised + body_shake)
          - G'azab     (Angry + body_shake + audio)
          - Jirkanish  (Disgust + torso_curl)

        Returns 9-tuple:
          (is_distress, face_score, pose_score, audio_score, motion_score,
           combined, emotions, dominant_emotion, face_is_fresh)
        """
        # ── 1. FACE score (distress emotionlar softmax sum) ──
        fer_cry, emotions, face_score, dominant_emotion, face_is_fresh = \
            self.emotion_analyser.analyse(frame, child_bbox=child_bbox)

        # ── 2. AUDIO score (FFT cry frequency) ──
        # Endi soft cry_score 0..1, combined logikaga TO'LIQ qo'shiladi
        audio_score = 0.0
        audio_enabled = self.audio_det._running
        if audio_enabled:
            audio_score = float(self.audio_det.cry_score)

        # ── 3. MOTION score (faqat audio yo'q bo'lsa fallback) ──
        # Audio bor bo'lsa, motion 0 — chunki audio aniqroq
        self.motion_det.update(frame)
        avg_motion = (sum(self.motion_det.history) / len(self.motion_det.history)
                      if self.motion_det.history else 0.0)
        motion_score = min(1.0, avg_motion / max(self.motion_det.threshold, 1e-5))

        # ── 4. WEIGHTED COMBINED ─────────────────────────────
        # Default: face=0.40, audio=0.35, pose=0.20, motion=0.05
        # Agar audio yo'q bo'lsa, audio vaznini motion ga o'tkazamiz
        if audio_enabled:
            combined = (face_score   * self.w_face +
                        audio_score  * self.w_audio +
                        pose_score   * self.w_pose +
                        motion_score * self.w_motion)
        else:
            # Audio yo'q — vazn motion ga o'tadi
            w_motion_eff = self.w_motion + self.w_audio
            combined = (face_score   * self.w_face +
                        pose_score   * self.w_pose +
                        motion_score * w_motion_eff)

        # ── 5. is_distress: multi-signal voting ──────────────
        # Strong signals = qaysi modallar yuqori ishonchda
        strong_signals = sum([
            face_score   >= self.strong_signal_thresh,
            audio_score  >= self.strong_signal_thresh,
            pose_score   >= self.strong_signal_thresh,
        ])

        is_distress = (
            combined >= self.combined_thresh        # weighted average yuqori
            or strong_signals >= 2                  # 2+ kuchli modallar
            or face_score  >= self.very_strong_thresh   # juda kuchli face
            or audio_score >= self.very_strong_thresh   # juda kuchli audio
        )

        return (is_distress, face_score, pose_score, audio_score, motion_score,
                combined, emotions, dominant_emotion, face_is_fresh)

    def _alert(self, atype, msg):
        now = time.time()
        if now - self.state.last_alert_time < self.cooldown: return
        self.state.last_alert_time = now
        self.state.alert_count += 1
        self.notifier.send(atype, msg)

    @staticmethod
    def _draw_skeleton(frame, kps, color=(0, 220, 220)):
        MIN_VIS = 0.3
        pts = {}
        for name, idx in KP.items():
            if idx < len(kps):
                x, y, v = kps[idx]
                if v >= MIN_VIS:
                    pts[name] = (int(x), int(y))
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        for a, b in SKELETON_BONES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color, 2)

    def _draw(self, frame, tracks,
              ch, ad, alone,
              is_distress, emotions, dominant_emotion,
              pose_signals, pose_score,
              face_score, pose_cry_score,
              audio_score, motion_score, combined_score,
              asec, cry_ratio,
              touch_score=0.0, touch_alerts=None, touch_contacts=None):
        h, w = frame.shape[:2]

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            color = (0, 60, 255) if alone else (0, 210, 70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            lbl = f"{'child' if t.height_ratio<self.adult_r else 'adult'} {t.height_ratio:.0%}"
            lbl_y = max(y1 - 10, 30)
            cv2.putText(frame, lbl, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 5)
            cv2.putText(frame, lbl, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            if t.kps is not None:
                sk_col = (0, 80, 255) if pose_score >= 0.4 else (0, 220, 220)
                self._draw_skeleton(frame, t.kps, sk_col)

        def put(txt, y, col=(240, 240, 240), fs=1.8, th=3):
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th+4)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)

        BAR_X, BAR_W, BAR_H = 10, 300, 22

        def score_bar(label, score, y, active_col, label_fs=1.3):
            cv2.putText(frame, label, (BAR_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_fs, (0,0,0), 5)
            cv2.putText(frame, label, (BAR_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_fs, (210,210,210), 2)
            bar_y = y + 6
            cv2.rectangle(frame, (BAR_X, bar_y), (BAR_X+BAR_W, bar_y+BAR_H), (50,50,50), -1)
            fill_w = int(BAR_W * min(score, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame, (BAR_X, bar_y), (BAR_X+fill_w, bar_y+BAR_H), active_col, -1)
            pct_txt = f"{score:.2f}"
            cv2.putText(frame, pct_txt, (BAR_X+BAR_W+10, bar_y+BAR_H-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 4)
            cv2.putText(frame, pct_txt, (BAR_X+BAR_W+10, bar_y+BAR_H-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230,230,230), 2)

        def divider(y):
            cv2.line(frame, (BAR_X, y), (BAR_X+BAR_W+80, y), (80,80,80), 1)

        put(f"Children:{ch}  Adults:{ad}", 55)
        alone_col = (30, 50, 255) if alone else (30, 210, 70)
        put("ALONE !" if alone else "SAFE", 115, alone_col)

        y_cur = 130
        if alone and asec > 0:
            pct = min(asec/self.alone_window, 1.0)
            cv2.rectangle(frame, (BAR_X, y_cur), (BAR_X+BAR_W, y_cur+20), (40,40,40), -1)
            cv2.rectangle(frame, (BAR_X, y_cur), (BAR_X+int(BAR_W*pct), y_cur+20), (0,130,255), -1)
            put(f"Alone:{asec:.0f}s/{self.alone_window}s", y_cur+50, (0,150,255), fs=1.3)
            y_cur += 80
        else:
            y_cur += 10

        divider(y_cur); y_cur += 30

        # Dominant emotion ko'rinishi (Sad/Fear/Angry/...)
        if self.emotion_analyser.enabled and emotions:
            dom_col = {
                "Sad":      (255, 140,  60),
                "Fear":     (200,  60, 200),
                "Angry":    ( 60,  60, 255),
                "Disgust":  ( 60, 180, 180),
                "Happy":    ( 60, 220, 100),
                "Surprise": (220, 220,  60),
                "Neutral":  (180, 180, 180),
            }.get(dominant_emotion, (200,200,60))
            dom_txt = f"EMOTION: {dominant_emotion}"
            cv2.putText(frame, dom_txt, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 5)
            cv2.putText(frame, dom_txt, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, dom_col, 2)
            y_cur += 40

            emo_str = "  ".join(f"{e}:{s:.0%}" for e, s in emotions[:2])
            cv2.putText(frame, emo_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
            cv2.putText(frame, emo_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,60), 2)
            y_cur += 40

        if pose_signals:
            sig_str = " | ".join(pose_signals)
            pose_detail_col = (40,100,255) if pose_score >= 0.4 else (160,160,60)
            cv2.putText(frame, sig_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 4)
            cv2.putText(frame, sig_str, (BAR_X, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, pose_detail_col, 2)
            y_cur += 42

        divider(y_cur); y_cur += 35

        FACE_COL   = (60, 180, 255)
        AUDIO_COL  = (255, 100, 100)
        POSE_COL   = (60, 255, 160)
        MOTION_COL = (180, 180, 60)

        score_bar("Face  ", face_score,   y_cur, FACE_COL);   y_cur += 70
        score_bar("Audio ", audio_score,  y_cur, AUDIO_COL);  y_cur += 70
        score_bar("Pose  ", pose_cry_score, y_cur, POSE_COL); y_cur += 70
        score_bar("Motion", motion_score, y_cur, MOTION_COL); y_cur += 70

        divider(y_cur); y_cur += 35

        COMBINED_COL = (0, 60, 255) if is_distress else (0, 180, 120)
        score_bar("COMBINED", combined_score, y_cur, COMBINED_COL, label_fs=1.5)
        cry_label = "  << DISTRESS" if is_distress else ""
        cry_lbl_col = (0,40,255) if is_distress else (100,200,100)
        cv2.putText(frame, cry_label, (BAR_X+BAR_W+80, y_cur+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 5)
        cv2.putText(frame, cry_label, (BAR_X+BAR_W+80, y_cur+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, cry_lbl_col, 2)
        y_cur += 75

        divider(y_cur); y_cur += 35

        score_bar("4min%  ", cry_ratio, y_cur, (0,80,255), label_fs=1.3)
        thr_x = BAR_X + int(BAR_W * self.cry_ratio_thresh)
        cv2.line(frame, (thr_x, y_cur+6), (thr_x, y_cur+6+BAR_H), (0,0,255), 3)
        cv2.putText(frame, f"thr:{self.cry_ratio_thresh:.0%}",
                    (thr_x-10, y_cur-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,60,255), 2)

        _ta = touch_alerts if touch_alerts is not None else []
        _tc = touch_contacts if touch_contacts is not None else []

        SENS_COLS = {"private":(0,30,255), "sensitive":(0,140,255), "normal":(80,200,80)}
        for t in tracks:
            if t.height_ratio < self.adult_r:
                bx1, by1, bx2, by2 = map(int, t.bbox)
                bh = by2 - by1
                for frac, lbl in [(0.25,"head"),(0.50,"chest"),(0.75,"groin")]:
                    zy = int(by1 + bh * frac)
                    col = SENS_COLS.get(ZONE_SENSITIVITY.get(lbl,"normal"), (120,120,120))
                    cv2.line(frame, (bx1, zy), (bx2, zy), col, 1)
                    cv2.putText(frame, lbl, (bx2+4, zy+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        for c in _tc:
            if "wrist" in c:
                wx, wy = int(c["wrist"][0]), int(c["wrist"][1])
                col = SENS_COLS.get(c["sensitivity"], (180,180,180))
                cv2.circle(frame, (wx, wy), 14, col, 3)
                cv2.circle(frame, (wx, wy), 4, (255,255,255), -1)
                cv2.putText(frame, c["zone"], (wx+16, wy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
                cv2.putText(frame, c["zone"], (wx+16, wy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        divider(y_cur); y_cur += 30

        TOUCH_COL = (0,40,255) if touch_score >= 0.5 else (
                     (0,140,255) if touch_score >= 0.25 else (60,200,60))
        score_bar("Touch ", touch_score, y_cur, TOUCH_COL, label_fs=1.3)
        y_cur += 65

        if _tc:
            for c in _tc[:3]:
                col = SENS_COLS.get(c["sensitivity"], (180,180,180))
                txt = f"  {c['zone']:12s} {c['duration']:4.1f}s  [{c['sensitivity']}]"
                cv2.putText(frame, txt, (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
                cv2.putText(frame, txt, (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
                y_cur += 38

        if TouchAlertState.EMERGENCY_ALERT in _ta:
            badge_col, badge_txt = (0,0,220), "!! TOUCH EMERGENCY !!"
        elif TouchAlertState.ALERT_ALERT in _ta:
            badge_col, badge_txt = (0,60,255), "! TOUCH ALERT !"
        elif TouchAlertState.WARN_ALERT in _ta:
            badge_col, badge_txt = (0,140,255), "TOUCH WARNING"
        else:
            badge_col = badge_txt = None

        if badge_txt:
            bx, by = 10, y_cur+10
            tw, th = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
            cv2.rectangle(frame, (bx-4, by-th-8), (bx+tw+8, by+6), badge_col, -1)
            cv2.putText(frame, badge_txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w-170, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (160,160,160), 2)
        return frame

    def run(self):
        cap = self._open_source()
        oc = self.cfg["output"]
        show = oc.get("show_video", True)
        save = oc.get("save_video", False)
        writer = None
        mc = self.cfg["model"]
        pcls = self.cfg["classes"]["person_class_id"]
        self.logger.info("Monitoring started. Press q to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame."); break

                h, w = frame.shape[:2]
                now = time.time()

                # YOLO
                res = self.model.predict(
                    frame,
                    conf=mc.get("confidence", 0.45),
                    iou=mc.get("iou", 0.65),
                    classes=[pcls],
                    imgsz=mc.get("imgsz", 640),
                    device=self.device,
                    verbose=False)

                tracks = self._extract_detections(res, h, w)
                ch, ad = self._classify_tracks(tracks)
                alone = self._is_alone(ch, ad)

                # Pose analysis — eng katta bola track
                pose_score, pose_signals = 0.0, []
                largest_child_kps = None
                largest_area = 0.0
                for t in tracks:
                    if t.height_ratio < self.adult_r and t.kps is not None:
                        x1, y1, x2, y2 = t.bbox
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            largest_child_kps = t.kps
                if largest_child_kps is not None:
                    pose_score, pose_signals = self.pose_analyser.analyse(largest_child_kps)

                child_bbox = self._pick_child_bbox(tracks)

                # MULTI-MODAL DISTRESS
                (is_distress, face_score, pose_cry_score, audio_score,
                 motion_score, combined_score, emotions,
                 dominant_emotion, face_is_fresh) = self._analyse_cry(
                    frame, pose_signals, pose_score, child_bbox=child_bbox)

                # Sliding window
                if alone:
                    self.cry_tracker.push(now, is_distress, is_fresh=face_is_fresh)
                else:
                    self.cry_tracker.reset()
                cry_ratio = self.cry_tracker.ratio()

                # Touch analysis
                child_distress_flag = is_distress or (pose_score >= 0.4)
                (touch_score, touch_alerts, touch_contacts) = self.touch_detector.analyse(
                    tracks, self.adult_r, child_distress_flag, now, h, w)

                for ta in touch_alerts:
                    top = self.touch_detector.top_contacts(1)
                    zone_info = top[0]["zone"] if top else "unknown"
                    dur_info = f"{top[0]['duration']:.1f}s" if top else ""
                    msgs = {
                        TouchAlertState.WARN_ALERT:
                            f"WARNING: Adult touching child ({zone_info}) for {dur_info}",
                        TouchAlertState.ALERT_ALERT:
                            f"ALERT: Inappropriate touching! Zone:{zone_info} {dur_info}",
                        TouchAlertState.EMERGENCY_ALERT:
                            f"EMERGENCY: Prolonged touch + distress! Zone:{zone_info} {dur_info}",
                    }
                    self._alert(ta, msgs.get(ta, ta))

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
                        # Distress sabablari
                        sources = []
                        if face_score  >= 0.4: sources.append(f"face={dominant_emotion}({face_score:.0%})")
                        if audio_score >= 0.3: sources.append(f"audio({audio_score:.0%})")
                        if pose_score  >= 0.4: sources.append(f"pose({','.join(pose_signals)})")
                        src_str = " | ".join(sources) or "motion"
                        self._alert(AlertState.CRY_ALERT,
                            f"EMERGENCY: Child DISTRESS {cry_ratio:.0%} of last "
                            f"{self.cry_window//60} min! [{src_str}]")
                else:
                    if self.state.alone_start is not None:
                        elapsed = now - self.state.alone_start
                        self.state.total_alone_seconds += elapsed
                        self.logger.info(f"Caregiver present. Alone: {elapsed:.0f}s")
                    self.state.alone_start = None

                if show or save:
                    vis = self._draw(frame.copy(), tracks,
                                     ch, ad, alone,
                                     is_distress, emotions, dominant_emotion,
                                     pose_signals, pose_score,
                                     face_score, pose_cry_score,
                                     audio_score, motion_score, combined_score,
                                     asec, cry_ratio,
                                     touch_score, touch_alerts, touch_contacts)
                    if show: cv2.imshow("Baby Monitor", vis)
                    if save:
                        if writer is None:
                            fps = oc.get("fps", 25)
                            res2 = tuple(oc.get("resolution", [w, h]))
                            writer = cv2.VideoWriter(oc["output_path"],
                                cv2.VideoWriter_fourcc(*"mp4v"), fps, res2)
                        writer.write(cv2.resize(vis, tuple(oc.get("resolution", [w, h]))))

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
