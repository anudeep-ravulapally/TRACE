# gait_utils.py
"""
TRACE Gait Recognition Utility
-------------------------------
Wraps the gait pipeline:
  video (.mp4) → YOLO silhouettes → GEI image → ResNet-18 embedding → cosine match

Designed to be imported into app.py exactly like face_utils.py.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ── Add gait module to path ────────────────────────────────────────────
GAIT_SRC = os.path.join(os.path.dirname(__file__), "modules", "gait", "src")
GAIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "modules", "gait", "models",
                                "baseline_gait_model.pth")
sys.path.insert(0, GAIT_SRC)
from modules.gait.src.phase3_dataset_and_model import BaselineGaitCNN

from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ── Constants ──────────────────────────────────────────────────────────
GAIT_THRESHOLD   = 0.50   # Checked against SCALED confidence, not raw score
GAIT_NUM_CLASSES = 74
GEI_SIZE         = (64, 64)

# Min-Max scaler bounds (calibrated from live debug scores)
BASE_MIN = 0.982   # Raw scores below this → crushed to 0.0
BASE_MAX = 1.000   # Theoretical upper bound

# ── Load model once at module import time ──────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = transforms.Compose([
    transforms.Resize(GEI_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

_gait_model = None

def _get_model():
    global _gait_model
    if _gait_model is None:
        model = BaselineGaitCNN(num_classes=GAIT_NUM_CLASSES)
        if os.path.exists(GAIT_MODEL_PATH):
            model.load_state_dict(
                torch.load(GAIT_MODEL_PATH, map_location=_device)
            )
        model.to(_device)
        model.eval()
        _gait_model = model
    return _gait_model


# ── Public API ─────────────────────────────────────────────────────────

def video_to_gei(video_path: str) -> np.ndarray:
    """
    Convert an .mp4 video to a Gait Energy Image (GEI).

    Uses YOLOv8-seg for person segmentation + silhouette averaging.
    Falls back to frame-differencing background subtraction if YOLO unavailable.

    Returns:
        numpy uint8 array of shape (64, 64) — the GEI
    Raises:
        ValueError if no person detected or video unreadable
    """
    try:
        from ultralytics import YOLO
        _yolo_video_to_gei(video_path)      # writes to tmp, returns path
    except ImportError:
        pass

    return _fallback_video_to_gei(video_path)


def _yolo_video_to_gei(video_path: str) -> np.ndarray:
    """YOLO-based GEI extraction (preferred)."""
    from ultralytics import YOLO
    yolo = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    silhouettes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, classes=[0], verbose=False, device=_device)
        for result in results:
            if result.masks is not None:
                mask = result.masks.data[0].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (mask * 255).astype(np.uint8)
                boxes = result.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    cropped = binary_mask[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, GEI_SIZE)
                    silhouettes.append(resized)
                break
    cap.release()

    if not silhouettes:
        raise ValueError("No person detected in video (YOLO).")

    gei = np.mean(np.array(silhouettes), axis=0).astype(np.uint8)
    return gei


def _fallback_video_to_gei(video_path: str) -> np.ndarray:
    """
    Fallback: MOG2 background subtraction → silhouette averaging → GEI.
    Works without YOLO/GPU.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=40, detectShadows=False
    )
    silhouettes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = fgbg.apply(gray)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                roi = fg_mask[y:y+h, x:x+w]
                resized = cv2.resize(roi, GEI_SIZE)
                silhouettes.append(resized)

    cap.release()

    if len(silhouettes) < 5:
        raise ValueError("Not enough movement detected in video. "
                         "Please record a 3-5 second walking clip.")

    gei = np.mean(np.array(silhouettes), axis=0).astype(np.uint8)
    return gei


def gei_to_embedding(gei: np.ndarray) -> list:
    """
    Convert a GEI numpy array (64×64 uint8) to a 512-d embedding vector.

    Returns:
        list of 512 floats
    """
    model = _get_model()
    pil_img = Image.fromarray(gei).convert("L")
    tensor = _transform(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        _, emb = model(tensor, return_embedding=True)

    return emb.cpu().numpy().flatten().tolist()


def get_gait_embedding_from_video(video_path: str) -> list:
    """
    Full pipeline: video path → 512-d gait embedding.

    Raises ValueError on failure.
    """
    gei = video_to_gei(video_path)
    return gei_to_embedding(gei)

def scale_gait_score(raw_score: float) -> float:
    """
    Min-Max Normalization "Confidence Punisher".
    """
    if raw_score < BASE_MIN:
        return 0.0
    scaled = (raw_score - BASE_MIN) / (BASE_MAX - BASE_MIN)
    return float(min(max(scaled, 0.0), 1.0))   # clamp for safety

def vanity_score(scaled: float) -> float:
    """
    Frontend vanity curve. Maps [0.50 → 1.0] to [0.85 → 0.99]
    """
    if scaled < GAIT_THRESHOLD:
        return scaled          # failed — don't dress up the score
    
    DISPLAY_LOW  = 0.85
    DISPLAY_HIGH = 0.99
    t = (scaled - GAIT_THRESHOLD) / (1.0 - GAIT_THRESHOLD)
    return DISPLAY_LOW + t * (DISPLAY_HIGH - DISPLAY_LOW)

def find_best_gait_match(new_embedding: list, all_users) -> tuple:
    """
    Compare gait embedding against all stored users using scaled confidence.
    Returns: (best_user, scaled_score, raw_score)
    """
    best_user      = None
    best_raw       = -1.0
    best_scaled    = 0.0

    new_vec = np.array(new_embedding).reshape(1, -1)

    for user in all_users:
        stored = user.get_gait_embedding()
        if stored is None:
            continue
        stored_vec = np.array(stored).reshape(1, -1)
        raw   = float(sk_cosine(new_vec, stored_vec)[0][0])
        scaled = scale_gait_score(raw)
        
        if scaled > best_scaled:
            best_raw    = raw
            best_scaled = scaled
            best_user   = user
        elif scaled == best_scaled and raw > best_raw:
            best_raw  = raw
            best_user = user

    if best_scaled >= GAIT_THRESHOLD:
        return best_user, best_scaled, best_raw
    
    return None, best_scaled, best_raw