# gait_utils.py
"""
TRACE Gait Recognition Utility
-------------------------------
Wraps the gait pipeline:
  video (.mp4) → YOLO silhouettes → GEI image → ResNet-18 embedding → cosine match

Designed to be imported into app.py exactly like face_utils.py.

Versioning
==========
This module is **config-driven** via a sidecar JSON next to the model file
(see :mod:`modules.gait.src.preproc_config`). Two pipelines are supported:

* **v1 (legacy)** — the original ``baseline_gait_model.pth`` trained with
  plain cross-entropy, square 64×64 GEIs, and the Min-Max "Confidence
  Punisher" scaling at inference. Activated when no sidecar config is found.

* **v2 (production)** — aspect-aware 64×44 GEIs, L2-normalized embeddings,
  raw cosine similarity matching with a calibrated threshold. Activated when
  the model ships a sidecar config (``model.config.json``) declaring
  ``score_scaling = "raw_cosine"``.

Test-time / inference improvements that are active in *both* paths:
  - flip TTA (inference embedding = mean of original + horizontal flip)
  - multi-clip enrollment: a user can store several clip embeddings; matching
    uses the **maximum** similarity across the clips (top-1 fusion)
  - vectorised gallery matching
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────
GAIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "modules", "gait", "models",
                                "baseline_gait_model.pth")

from modules.gait.src.phase3_dataset_and_model import BaselineGaitCNN
from modules.gait.src.preproc_config import load_gait_config
from modules.gait.src import preprocessing as _pre

from torchvision import transforms

# ── Config (sidecar JSON + legacy v1 fallback) ─────────────────────────
_CONFIG = load_gait_config(GAIT_MODEL_PATH)

# Public constants — kept on the module for backward compat. They reflect
# whichever pipeline (v1 / v2) the loaded model corresponds to.
GAIT_THRESHOLD   = _CONFIG.match_threshold
GAIT_NUM_CLASSES = _CONFIG.num_classes
GEI_SIZE         = tuple(_CONFIG.gei_size)            # (H, W)

# Min-Max scaler bounds — only used on the v1 legacy path.
BASE_MIN = _CONFIG.minmax_base_min
BASE_MAX = _CONFIG.minmax_base_max

_USE_RAW_COSINE = (_CONFIG.score_scaling == "raw_cosine")
_L2_NORMALIZE   = bool(_CONFIG.l2_normalize_embeddings)

# ── Load model once at module import time ──────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = transforms.Compose([
    transforms.Resize(GEI_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(_CONFIG.normalize_mean),
                         std=list(_CONFIG.normalize_std)),
])

_gait_model = None
_yolo_model = None


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


def _get_yolo():
    """Lazy-load the YOLOv8-seg model exactly once per process.

    Returns the model instance, or None if ultralytics/YOLO weights are not
    available (caller should fall back to the MOG2 path).
    """
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    try:
        _yolo_model = YOLO("yolov8n-seg.pt")
    except Exception:
        # Model file missing or load failure — leave cache as None so we don't
        # retry-load every call, but only for this process lifetime.
        return None
    return _yolo_model


# ── Public API ─────────────────────────────────────────────────────────

def video_to_gei(video_path: str) -> np.ndarray:
    """
    Convert an .mp4 video to a Gait Energy Image (GEI).

    Uses YOLOv8-seg for person segmentation + silhouette averaging.
    Falls back to MOG2 background subtraction if YOLO is unavailable or
    fails to detect any person in the video.

    Returns:
        numpy uint8 array of shape ``GEI_SIZE`` (H, W) — the GEI
    Raises:
        ValueError if no person detected or video unreadable
    """
    yolo = _get_yolo()
    if yolo is not None:
        try:
            return _yolo_video_to_gei(video_path, yolo)
        except ValueError:
            # No person detected via YOLO — try fallback before giving up.
            pass

    return _fallback_video_to_gei(video_path)


def _yolo_video_to_gei(video_path: str, yolo=None) -> np.ndarray:
    """YOLO-based GEI extraction (preferred).

    On the v2 pipeline: picks the largest/most-confident person mask, applies
    silhouette QC, aspect-aware centroid alignment, and period-aware GEI
    averaging.

    On v1: falls back to the original square-resize behaviour so the existing
    checkpoint's expected input geometry is preserved.
    """
    if yolo is None:
        yolo = _get_yolo()
        if yolo is None:
            raise ValueError("YOLO model unavailable.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    silhouettes = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo(frame, classes=[0], verbose=False, device=_device)
            for result in results:
                if _CONFIG.aspect_aware_align:
                    # v2 path: best-mask → QC → aspect-aware align.
                    full_mask = _pre.silhouette_from_yolo_result(result, frame.shape[:2])
                    if full_mask is None:
                        break
                    aligned = _pre.aligned_silhouette_from_mask(
                        full_mask, out_size=GEI_SIZE,
                    )
                    if aligned is not None:
                        silhouettes.append(aligned)
                else:
                    # v1 path: byte-for-byte the original behaviour.
                    if result.masks is not None:
                        mask = result.masks.data[0].cpu().numpy()
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        binary_mask = (mask * 255).astype(np.uint8)
                        boxes = result.boxes.xyxy.cpu().numpy()
                        if len(boxes) > 0:
                            x1, y1, x2, y2 = map(int, boxes[0])
                            cropped = binary_mask[y1:y2, x1:x2]
                            if cropped.size == 0:
                                break
                            resized = cv2.resize(cropped, (GEI_SIZE[1], GEI_SIZE[0]))
                            silhouettes.append(resized)
                break
    finally:
        cap.release()

    if len(silhouettes) < _CONFIG.min_silhouette_frames:
        raise ValueError("No person detected in video (YOLO).")

    if _CONFIG.aspect_aware_align:
        gei = _pre.build_gei(silhouettes, use_period_detection=_CONFIG.use_period_detection)
        if gei is None:
            raise ValueError("Failed to build GEI.")
        return gei
    return np.mean(np.array(silhouettes), axis=0).astype(np.uint8)


def _fallback_video_to_gei(video_path: str) -> np.ndarray:
    """
    Fallback: MOG2 background subtraction → silhouette averaging → GEI.
    Works without YOLO/GPU.

    Note: MOG2 silhouettes are visually different from YOLO/CASIA-B silhouettes,
    so embeddings from this path are less reliable — kept as a "better than
    nothing" path when YOLO isn't available.
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
                if _CONFIG.aspect_aware_align:
                    if _pre.silhouette_is_acceptable(roi):
                        aligned = _pre.align_silhouette(roi, out_size=GEI_SIZE)
                        if aligned is not None:
                            silhouettes.append(aligned)
                else:
                    resized = cv2.resize(roi, (GEI_SIZE[1], GEI_SIZE[0]))
                    silhouettes.append(resized)

    cap.release()

    if len(silhouettes) < max(5, _CONFIG.min_silhouette_frames):
        raise ValueError("Not enough movement detected in video. "
                         "Please record a 3-5 second walking clip.")

    if _CONFIG.aspect_aware_align:
        gei = _pre.build_gei(silhouettes, use_period_detection=_CONFIG.use_period_detection)
        if gei is None:
            raise ValueError("Failed to build GEI.")
        return gei
    return np.mean(np.array(silhouettes), axis=0).astype(np.uint8)


def gei_to_embedding(gei: np.ndarray, flip_tta: bool = True) -> list:
    """
    Convert a GEI numpy array to a 512-d embedding vector.

    Args:
        gei: HxW uint8 GEI image.
        flip_tta: If True (default) average the embedding of the GEI and its
            horizontal flip — a free ~0.5–1pp accuracy bump for symmetric gait
            patterns. Disable to recover the legacy single-pass behaviour.

    Returns:
        list of 512 floats. L2-normalized when the loaded model's config says
        embeddings should be normalized; raw otherwise.
    """
    model = _get_model()
    pil_img = Image.fromarray(gei).convert("L")
    tensor = _transform(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        _, emb = model(tensor, return_embedding=True)
        if flip_tta:
            tensor_f = torch.flip(tensor, dims=[-1])
            _, emb_f = model(tensor_f, return_embedding=True)
            emb = (emb + emb_f) * 0.5

        if _L2_NORMALIZE:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

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
    Min-Max Normalization "Confidence Punisher" — legacy v1 only.

    On v2 (raw-cosine) models this is the identity (clamped to [0, 1]) so the
    function remains safe to call from any callsite.
    """
    if _USE_RAW_COSINE:
        return float(min(max(raw_score, 0.0), 1.0))
    if raw_score < BASE_MIN:
        return 0.0
    scaled = (raw_score - BASE_MIN) / (BASE_MAX - BASE_MIN)
    return float(min(max(scaled, 0.0), 1.0))   # clamp for safety

def vanity_score(scaled: float) -> float:
    """
    Frontend vanity curve. Maps [threshold → 1.0] to [0.85 → 0.99].

    On v2 raw-cosine models the scaled score already lives on [0, 1] in a
    well-distributed way, so the same vanity remap still produces sensible
    UI confidence values.
    """
    if scaled < GAIT_THRESHOLD:
        return scaled          # failed — don't dress up the score

    DISPLAY_LOW  = 0.85
    DISPLAY_HIGH = 0.99
    t = (scaled - GAIT_THRESHOLD) / (1.0 - GAIT_THRESHOLD)
    return DISPLAY_LOW + t * (DISPLAY_HIGH - DISPLAY_LOW)


# ── Matching ───────────────────────────────────────────────────────────
def _user_clip_embeddings(user) -> list:
    """Return a (possibly empty) list of stored clip embeddings for a user.

    Uses ``get_gait_embeddings`` (plural, multi-clip) if the User model
    exposes it, otherwise falls back to the single-clip ``get_gait_embedding``.
    """
    fn = getattr(user, "get_gait_embeddings", None)
    if callable(fn):
        clips = fn() or []
        return [c for c in clips if c]
    single = user.get_gait_embedding()
    return [single] if single else []


def find_best_gait_match(new_embedding: list, all_users) -> tuple:
    """
    Compare a probe embedding against all stored users and return
    ``(best_user, scaled_score, raw_score)``.

    Behaviour:
      - **Multi-clip galleries**: each user contributes one or more enrollment
        clip embeddings. The user's similarity is the **maximum** cosine
        across their clips (top-1 score fusion).
      - **v2 (raw-cosine) models**: the threshold operates directly on cosine.
      - **v1 (legacy) models**: the Min-Max "Confidence Punisher" is applied
        to preserve previously-calibrated thresholds.

    Vectorised: stacks every clip from every user into one matrix, computes
    similarity in a single matmul, then reduces per-user with a max.
    """
    candidate_users = []
    candidate_vecs = []           # flat list of all clip vectors
    clip_to_user_idx = []         # parallel: index into candidate_users
    for user in all_users:
        clips = _user_clip_embeddings(user)
        if not clips:
            continue
        u_idx = len(candidate_users)
        candidate_users.append(user)
        for c in clips:
            candidate_vecs.append(np.asarray(c, dtype=np.float32))
            clip_to_user_idx.append(u_idx)

    if not candidate_users:
        return None, 0.0, -1.0

    new_vec = np.asarray(new_embedding, dtype=np.float32).ravel()
    stored_mat = np.vstack(candidate_vecs)  # (N_clips, D)

    new_norm = np.linalg.norm(new_vec)
    stored_norms = np.linalg.norm(stored_mat, axis=1)
    eps = 1e-12
    raw_clip_scores = (stored_mat @ new_vec) / (stored_norms * new_norm + eps)

    # Reduce: per-user → max similarity across that user's clips.
    # (Tracking by integer index, not id(), so the result is stable even if
    #  the ORM rebuilds user objects mid-call.)
    raw_scores = np.full(len(candidate_users), -np.inf, dtype=np.float32)
    np.maximum.at(raw_scores, clip_to_user_idx, raw_clip_scores)

    if _USE_RAW_COSINE:
        # Cosine ∈ [-1, 1]; clamp negative values to 0 for the scaled view.
        scaled_scores = np.clip(raw_scores, 0.0, 1.0)
    else:
        denom = (BASE_MAX - BASE_MIN)
        scaled_scores = np.clip((raw_scores - BASE_MIN) / denom, 0.0, 1.0)
        scaled_scores = np.where(raw_scores < BASE_MIN, 0.0, scaled_scores)

    # Pick best by scaled score, breaking ties by raw score.
    order = np.lexsort((raw_scores, scaled_scores))
    best_idx = int(order[-1])
    best_raw = float(raw_scores[best_idx])
    best_scaled = float(scaled_scores[best_idx])
    best_user = candidate_users[best_idx]

    if best_scaled >= GAIT_THRESHOLD:
        return best_user, best_scaled, best_raw

    return None, best_scaled, best_raw
