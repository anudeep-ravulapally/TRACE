"""
Step 0 — CASIA-B evaluation harness.

This is the production gate. Without it, every other accuracy claim is a
guess. It implements the standard CASIA-B evaluation protocol:

  * **Train / Test split** : subjects 001..074 train, 075..124 test.
  * **Gallery**             : within the test set, conditions ``nm-01..nm-04``.
  * **Probes**              : within the test set,
                              ``nm-05, nm-06`` (Normal),
                              ``bg-01, bg-02`` (Bag),
                              ``cl-01, cl-02`` (Coat).

Reports:
  * **Rank-1 identification accuracy** per condition (NM / BG / CL),
    optionally broken down per probe angle when the GEI filenames encode angle.
  * **Verification ROC** : TAR @ FAR = 1e-2 and 1e-3, plus the AUC, computed
    over all (probe, gallery) pairs as a binary genuine/imposter problem.

The harness loads the model + sidecar config via :mod:`preproc_config` so the
metrics correspond exactly to the (model, preproc) pair shipped with the
checkpoint. Identification scores naturally drop the legacy "Confidence
Punisher" — verification thresholds are derived directly from the ROC curve.

Usage::

    python -m modules.gait.src.evaluate \\
        --data ./dataset/GEI_Data \\
        --model ./modules/gait/models/baseline_gait_model.pth
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make this module runnable both as `-m` and as a plain script.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Heavy imports kept inside main() so `--help` stays cheap.

# Standard CASIA-B protocol.
TRAIN_SUBJECTS = [f"{i:03d}" for i in range(1, 75)]
TEST_SUBJECTS = [f"{i:03d}" for i in range(75, 125)]
GALLERY_CONDITIONS = ("nm-01", "nm-02", "nm-03", "nm-04")
PROBE_CONDITIONS_BY_GROUP: Dict[str, Tuple[str, ...]] = {
    "NM": ("nm-05", "nm-06"),
    "BG": ("bg-01", "bg-02"),
    "CL": ("cl-01", "cl-02"),
}

# Filenames in CASIA-B GEI dumps usually encode the camera angle as e.g. "090.png"
# or "..._090.png". This regex pulls out the leading 3-digit angle if present.
_ANGLE_RE = re.compile(r"(?:^|[_-])(\d{3})\.png$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------
def _angle_from_path(path: Path) -> Optional[int]:
    m = _ANGLE_RE.search(path.name)
    return int(m.group(1)) if m else None


def collect_gei_paths(
    data_dir: Path,
    subjects: Sequence[str],
    conditions: Sequence[str],
) -> List[Tuple[Path, str, str, Optional[int]]]:
    """Return tuples of (path, subject, condition, angle) for matching GEIs."""
    out: List[Tuple[Path, str, str, Optional[int]]] = []
    for subj in subjects:
        sd = data_dir / subj
        if not sd.is_dir():
            continue
        for cond in conditions:
            cd = sd / cond
            if not cd.is_dir():
                continue
            for img in sorted(cd.glob("*.png")):
                out.append((img, subj, cond, _angle_from_path(img)))
    return out


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def _build_transform(config):
    from torchvision import transforms

    h, w = config.gei_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(config.normalize_mean),
                             std=list(config.normalize_std)),
    ])


def _embed_paths(model, paths: Sequence[Path], transform, device, l2_normalize: bool,
                 batch_size: int = 64) -> np.ndarray:
    import torch
    from PIL import Image

    embs: List[np.ndarray] = []
    batch_imgs = []

    def _flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            _, emb = model(x, return_embedding=True)
        if l2_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu().numpy().astype(np.float32))
        batch_imgs.clear()

    for p in paths:
        img = Image.open(p).convert("L")
        batch_imgs.append(transform(img))
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()
    return np.vstack(embs) if embs else np.zeros((0, 512), dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) × (M, D) → (N, M) cosine similarity."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def rank1_identification(
    probe_emb: np.ndarray,
    probe_subjects: Sequence[str],
    gallery_emb: np.ndarray,
    gallery_subjects: Sequence[str],
) -> float:
    """Closed-set Rank-1 over the test gallery (gallery is one vector per
    (subject, gallery_condition, angle) — we average to one per subject below
    if the caller wants per-subject gallery)."""
    if probe_emb.size == 0 or gallery_emb.size == 0:
        return float("nan")
    sim = _cosine_similarity(probe_emb, gallery_emb)
    pred_idx = np.argmax(sim, axis=1)
    pred_subj = np.array([gallery_subjects[i] for i in pred_idx])
    correct = (pred_subj == np.asarray(probe_subjects)).sum()
    return float(correct) / float(len(probe_subjects))


def per_subject_gallery(
    gallery_emb: np.ndarray,
    gallery_subjects: Sequence[str],
    l2_normalize: bool,
) -> Tuple[np.ndarray, List[str]]:
    """Mean-pool gallery embeddings to one vector per subject."""
    by_subj: Dict[str, List[np.ndarray]] = defaultdict(list)
    for vec, subj in zip(gallery_emb, gallery_subjects):
        by_subj[subj].append(vec)
    subjects = sorted(by_subj.keys())
    means = np.vstack([np.mean(np.vstack(by_subj[s]), axis=0) for s in subjects])
    if l2_normalize:
        means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    return means.astype(np.float32), subjects


def verification_metrics(
    probe_emb: np.ndarray,
    probe_subjects: Sequence[str],
    gallery_emb: np.ndarray,
    gallery_subjects: Sequence[str],
) -> Dict[str, float]:
    """Compute open-set verification metrics over all (probe, gallery) pairs.

    A pair is *genuine* iff the subject IDs match. Returns:
        - auc                  : ROC area-under-curve.
        - tar_at_far_1e-2     : True-accept rate at FAR = 1e-2.
        - tar_at_far_1e-3     : True-accept rate at FAR = 1e-3.
    """
    if probe_emb.size == 0 or gallery_emb.size == 0:
        return {"auc": float("nan"), "tar_at_far_1e-2": float("nan"),
                "tar_at_far_1e-3": float("nan")}

    sim = _cosine_similarity(probe_emb, gallery_emb).ravel()
    probe_subj_arr = np.asarray(probe_subjects)
    gallery_subj_arr = np.asarray(gallery_subjects)
    genuine = (probe_subj_arr[:, None] == gallery_subj_arr[None, :]).ravel()

    pos = sim[genuine]
    neg = sim[~genuine]
    if pos.size == 0 or neg.size == 0:
        return {"auc": float("nan"), "tar_at_far_1e-2": float("nan"),
                "tar_at_far_1e-3": float("nan")}

    # FAR-target → score threshold from sorted negatives, then TAR.
    def tar_at(far_target: float) -> float:
        sorted_neg = np.sort(neg)
        # The threshold above which fraction == far_target negatives are accepted.
        k = int(np.ceil((1.0 - far_target) * len(sorted_neg))) - 1
        k = max(0, min(len(sorted_neg) - 1, k))
        thr = sorted_neg[k]
        return float((pos > thr).mean())

    # AUC via Mann-Whitney U (no sklearn dependency required).
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    auc = (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (
        len(pos) * len(neg)
    )

    return {
        "auc": float(auc),
        "tar_at_far_1e-2": tar_at(1e-2),
        "tar_at_far_1e-3": tar_at(1e-3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a gait model on CASIA-B.")
    p.add_argument("--data", required=True, help="GEI data root (subject/condition/angle.png).")
    p.add_argument("--model", required=True, help="Model checkpoint (.pth).")
    p.add_argument("--per-angle", action="store_true",
                   help="Also break Rank-1 down by probe angle.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch

    from modules.gait.src.phase3_dataset_and_model import BaselineGaitCNN
    from modules.gait.src.preproc_config import load_gait_config

    config = load_gait_config(args.model)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    print(f"=== Evaluating {args.model}")
    print(f"    model_version   : {config.model_version}")
    print(f"    preproc_version : {config.preproc_version}")
    print(f"    gei_size        : {config.gei_size}")
    print(f"    l2_normalize    : {config.l2_normalize_embeddings}")

    model = BaselineGaitCNN(num_classes=config.num_classes)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    transform = _build_transform(config)
    data_dir = Path(args.data)

    # Gallery — one vector per (subject, gallery condition, angle), then mean-pool per subject.
    g_items = collect_gei_paths(data_dir, TEST_SUBJECTS, GALLERY_CONDITIONS)
    if not g_items:
        print(f"!! No gallery GEIs found under {data_dir}. "
              f"Expected subjects 075..124 with conditions {GALLERY_CONDITIONS}.")
        return 1
    g_paths = [p for p, _, _, _ in g_items]
    g_subjects = [s for _, s, _, _ in g_items]
    print(f"    gallery items   : {len(g_items)}")
    g_emb_raw = _embed_paths(model, g_paths, transform, device,
                             l2_normalize=config.l2_normalize_embeddings)
    g_emb_subj, g_subj_list = per_subject_gallery(
        g_emb_raw, g_subjects, l2_normalize=config.l2_normalize_embeddings,
    )
    print(f"    gallery subjects: {len(g_subj_list)}")

    print()
    print(f"{'Group':<6} {'Probe':<6} {'Items':>6}  {'Rank-1':>8}")
    overall = {}
    for group, conds in PROBE_CONDITIONS_BY_GROUP.items():
        items = collect_gei_paths(data_dir, TEST_SUBJECTS, conds)
        if not items:
            print(f"{group:<6} {'-':<6} {0:>6}  {'(no data)':>8}")
            continue
        paths = [p for p, _, _, _ in items]
        subjects = [s for _, s, _, _ in items]
        angles = [a for _, _, _, a in items]
        emb = _embed_paths(model, paths, transform, device,
                           l2_normalize=config.l2_normalize_embeddings)
        r1 = rank1_identification(emb, subjects, g_emb_subj, g_subj_list)
        print(f"{group:<6} {','.join(conds):<6} {len(items):>6}  {r1*100:>7.2f}%")
        overall[group] = (emb, subjects, angles, r1)

        if args.per_angle:
            by_angle: Dict[int, List[int]] = defaultdict(list)
            for i, a in enumerate(angles):
                if a is not None:
                    by_angle[a].append(i)
            for ang in sorted(by_angle.keys()):
                idx = by_angle[ang]
                sub_r1 = rank1_identification(
                    emb[idx], [subjects[i] for i in idx],
                    g_emb_subj, g_subj_list,
                )
                print(f"  └ {ang:03d}°  n={len(idx):>4}  "
                      f"Rank-1={sub_r1*100:.2f}%")

    # Verification — pool all probes against the per-subject gallery.
    print()
    if overall:
        all_probe_emb = np.vstack([e for e, *_ in overall.values()])
        all_probe_subj = []
        for _, s, *_ in overall.values():
            all_probe_subj.extend(s)
        v = verification_metrics(all_probe_emb, all_probe_subj,
                                 g_emb_subj, g_subj_list)
        print(f"Verification (all probes vs per-subject gallery):")
        print(f"  AUC                  : {v['auc']*100:>6.2f}%")
        print(f"  TAR @ FAR=1e-2       : {v['tar_at_far_1e-2']*100:>6.2f}%")
        print(f"  TAR @ FAR=1e-3       : {v['tar_at_far_1e-3']*100:>6.2f}%")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
