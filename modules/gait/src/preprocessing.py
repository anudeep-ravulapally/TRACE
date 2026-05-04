"""
v2 silhouette / GEI preprocessing.

This module is what Step 1 of the accuracy plan calls for:

1. **Aspect-aware alignment.** Pad to a target aspect ratio and resize to a
   non-square output (default 64×44, the GaitSet/GEINet standard) instead of
   squashing every silhouette to 64×64.
2. **Best-mask selection.** Pick the largest / most-confident person mask in
   each frame instead of blindly grabbing index 0.
3. **Silhouette QC.** Discard frames with implausibly small foregrounds,
   ratios outside a sensible band, or fragmented masks.
4. **Period-aware GEI.** Detect a gait cycle from silhouette-width oscillation
   and average ~one full cycle from the middle of the clip.
5. A clean fallback path with the same QC, so MOG2 and YOLO produce
   consistent silhouette geometry.

The functions here are pure (numpy / OpenCV only) — torch is not imported so
this module is cheap to use from training, inference, and tests alike.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def silhouette_is_acceptable(
    mask: np.ndarray,
    *,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 0.95,
    max_components: int = 2,
) -> bool:
    """Return True if a binary silhouette looks like a complete walking person.

    ``mask`` is a 2D uint8 array of the *cropped* silhouette (foreground = >0).
    Three cheap checks:
      - foreground area / bbox area is in [min_area_ratio, max_area_ratio]
        (rejects tiny specks and full-frame blobs caused by camera shake)
      - at most ``max_components`` connected foreground components
        (rejects fragmented masks)
      - bbox aspect (h / w) is between 1.0 and 4.0 — people are taller than
        they are wide; horizontal blobs are almost always not a person
    """
    if mask is None or mask.size == 0:
        return False

    h, w = mask.shape[:2]
    if h < 8 or w < 4:
        return False

    fg = (mask > 0).astype(np.uint8)
    area = int(fg.sum())
    bbox = float(h * w)
    ratio = area / bbox if bbox > 0 else 0.0
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return False

    aspect = h / max(w, 1)
    if aspect < 1.0 or aspect > 4.0:
        return False

    # Connected components — cheap on small crops.
    n_labels, _ = cv2.connectedComponents(fg)
    # n_labels includes the background label, so subtract 1.
    if (n_labels - 1) > max_components:
        return False

    return True


# ---------------------------------------------------------------------------
# Aspect-aware alignment
# ---------------------------------------------------------------------------
def align_silhouette(
    mask: np.ndarray,
    out_size: Tuple[int, int] = (64, 44),
) -> Optional[np.ndarray]:
    """Centroid-align a binary silhouette onto a fixed-size canvas.

    Steps (this is the standard CASIA-B / GaitSet preprocessing):
      1. Tight-crop the foreground bounding box.
      2. Scale so the silhouette height matches ``out_size[0]`` and width
         is preserved by the same scale (no aspect distortion).
      3. Paste onto an ``out_size`` canvas with horizontal placement chosen
         so that the silhouette's *centroid* lands at the canvas centre.
         Pixels falling outside the canvas (very wide silhouettes) are clipped.

    Returns a ``uint8`` array of shape ``out_size`` (H, W), or ``None`` if the
    input is empty.
    """
    if mask is None or mask.size == 0:
        return None

    out_h, out_w = out_size
    fg = (mask > 0).astype(np.uint8) * 255

    ys, xs = np.where(fg > 0)
    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = fg[y0:y1, x0:x1]

    ch, cw = cropped.shape
    if ch <= 0 or cw <= 0:
        return None

    # Scale by height; preserve aspect.
    scale = out_h / float(ch)
    new_w = max(1, int(round(cw * scale)))
    resized = cv2.resize(cropped, (new_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Centroid X within the resized silhouette.
    rs_ys, rs_xs = np.where(resized > 0)
    if rs_xs.size == 0:
        return None
    cx_resized = float(rs_xs.mean())

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    # Place such that the centroid sits at canvas centre.
    target_cx = out_w / 2.0
    left = int(round(target_cx - cx_resized))

    # Compute clipped paste region.
    src_x0 = max(0, -left)
    src_x1 = min(new_w, out_w - left)
    dst_x0 = max(0, left)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_x1 > src_x0 and dst_x1 > dst_x0:
        canvas[:, dst_x0:dst_x1] = resized[:, src_x0:src_x1]

    return canvas


# ---------------------------------------------------------------------------
# Best-mask picking from a multi-detection result
# ---------------------------------------------------------------------------
def pick_best_person_mask(
    masks: Sequence[np.ndarray],
    boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    confidences: Optional[Sequence[float]] = None,
) -> Optional[int]:
    """Choose the index of the most plausible single-person mask.

    Strategy: rank by ``confidence * bbox_area`` (or just bbox_area if
    confidences are unavailable), which favours large, confident detections
    over distant or low-confidence ones.

    Returns the chosen index, or ``None`` if ``masks`` is empty.
    """
    if masks is None or len(masks) == 0:
        return None

    n = len(masks)
    areas = np.zeros(n, dtype=np.float64)
    if boxes_xyxy is not None and len(boxes_xyxy) == n:
        for i, b in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            areas[i] = max(0.0, (x2 - x1) * (y2 - y1))
    else:
        for i, m in enumerate(masks):
            areas[i] = float((m > 0).sum())

    if confidences is not None and len(confidences) == n:
        scores = areas * np.asarray(confidences, dtype=np.float64)
    else:
        scores = areas

    if not np.any(scores > 0):
        return None
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Period-aware GEI from a stack of aligned silhouettes
# ---------------------------------------------------------------------------
def estimate_gait_period(silhouettes: Sequence[np.ndarray]) -> Optional[int]:
    """Estimate gait cycle length (in frames) from silhouette width oscillation.

    Walking causes the silhouette's foreground width to oscillate as the legs
    swing in/out. The autocorrelation of that signal peaks at integer
    multiples of the gait period.

    Returns the period in frames, or ``None`` if the signal is too short / flat.
    Periods outside [10, 60] frames (well outside a normal walking cycle at
    typical 24-60fps cameras) are rejected as noise.
    """
    n = len(silhouettes)
    if n < 20:
        return None

    widths = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(silhouettes):
        cols_with_fg = np.where(s.sum(axis=0) > 0)[0]
        # +1 so a single-column silhouette has width 1, not 0.
        widths[i] = float(cols_with_fg[-1] - cols_with_fg[0] + 1) if cols_with_fg.size else 0.0

    sig = widths - widths.mean()
    if np.allclose(sig, 0):
        return None

    # Normalised autocorrelation up to half the sequence length.
    full = np.correlate(sig, sig, mode="full")
    ac = full[full.size // 2:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    # Search for the first peak after lag=1 within reasonable cycle range.
    lo, hi = 10, min(60, len(ac) - 1)
    if hi <= lo:
        return None
    # A "peak": local max greater than its neighbours and above a small threshold.
    best_lag = None
    best_val = 0.2  # require at least mild periodicity
    for lag in range(lo, hi):
        if ac[lag] > best_val and ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            best_val = ac[lag]
            best_lag = lag
            break
    return best_lag


def build_gei(
    silhouettes: Sequence[np.ndarray],
    use_period_detection: bool = True,
) -> Optional[np.ndarray]:
    """Average a stack of aligned silhouettes into a Gait Energy Image.

    If ``use_period_detection`` and a gait period is detectable, average over
    one cycle centred on the middle of the clip — this removes phase bias from
    clips of different lengths. Otherwise fall back to averaging everything.
    """
    if silhouettes is None or len(silhouettes) == 0:
        return None

    arr = np.asarray(silhouettes, dtype=np.float32)
    n = arr.shape[0]

    if use_period_detection:
        period = estimate_gait_period(silhouettes)
        if period is not None and period < n:
            mid = n // 2
            half = period // 2
            lo = max(0, mid - half)
            hi = min(n, lo + period)
            arr = arr[lo:hi]

    return arr.mean(axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# End-to-end helpers used by inference and the GEI extraction script.
# ---------------------------------------------------------------------------
def silhouette_from_yolo_result(
    result,
    frame_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Turn a single ``ultralytics`` YOLO result into a binary silhouette mask
    at the original frame's resolution, picking the best person.

    Returns a uint8 (H, W) array or None if no acceptable mask is present.
    """
    if result is None or getattr(result, "masks", None) is None:
        return None

    masks_t = result.masks.data
    if masks_t is None or len(masks_t) == 0:
        return None

    # Convert to numpy lazily (these can be torch tensors).
    masks_np: List[np.ndarray] = []
    for m in masks_t:
        try:
            masks_np.append(m.cpu().numpy())
        except AttributeError:
            masks_np.append(np.asarray(m))

    boxes = None
    confidences = None
    if getattr(result, "boxes", None) is not None:
        try:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
        except Exception:
            boxes = None
        try:
            confidences = result.boxes.conf.cpu().numpy().tolist()
        except Exception:
            confidences = None

    idx = pick_best_person_mask(masks_np, boxes_xyxy=boxes, confidences=confidences)
    if idx is None:
        return None

    mask = masks_np[idx]
    h, w = frame_shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask * 255).astype(np.uint8)


def crop_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight-crop a binary mask to its foreground bounding box."""
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return mask[y0:y1, x0:x1]


def aligned_silhouette_from_mask(
    mask: np.ndarray,
    out_size: Tuple[int, int] = (64, 44),
) -> Optional[np.ndarray]:
    """Convenience: crop → QC → aspect-aware align."""
    cropped = crop_to_bbox(mask)
    if cropped is None:
        return None
    if not silhouette_is_acceptable(cropped):
        return None
    return align_silhouette(cropped, out_size=out_size)
