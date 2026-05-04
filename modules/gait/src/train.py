"""
Step 2 — Metric-learning training script for the gait backbone.

This script replaces the implicit "train ResNet18 with cross-entropy" workflow
with a properly-supervised metric-learning loop. Two losses are supported:

  * ``--loss arcface`` : ArcFace head + cross-entropy. Closed-set accuracy on
    the 74 train subjects guides training; embeddings are L2-normalized and
    open-set generalization comes from the angular margin.

  * ``--loss triplet`` : Batch-hard triplet loss with a P×K sampler. No
    classifier head; the embedding space is shaped directly.

Usage::

    python -m modules.gait.src.train \\
        --data ./dataset/GEI_Data \\
        --out  ./modules/gait/models/gait_v2.pth \\
        --loss arcface --epochs 60

A sidecar config (``gait_v2.config.json``) is written next to the checkpoint
so that ``gait_utils`` automatically picks up the new preprocessing /
inference settings — see :mod:`modules.gait.src.preproc_config`.

Architecture upgrades (GaitSet / GaitPart) are an intentional *next* step:
the ``--arch`` switch leaves a hook for it but only ResNet18 is implemented
here. Hyperparameter tuning (Optuna) wraps this script as the inner loop and
is left for a follow-up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Make this module runnable both as `python -m modules.gait.src.train` and
# `python modules/gait/src/train.py`.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.gait.src.phase3_dataset_and_model import (  # noqa: E402
    BaselineGaitCNN,
    CASIABDataset,
)
from modules.gait.src.losses import ArcFaceHead, BatchHardTripletLoss  # noqa: E402
from modules.gait.src.sampler import PKBatchSampler, labels_from_dataset  # noqa: E402
from modules.gait.src.preproc_config import GaitConfig, sidecar_path_for  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a metric-learning gait model.")
    p.add_argument("--data", required=True, help="Path to GEI data root (subject/condition/angle.png)")
    p.add_argument("--out", required=True, help="Path to write the trained .pth")
    p.add_argument("--arch", default="resnet18", choices=["resnet18"],
                   help="Backbone architecture. GaitSet / GaitPart are TODO.")
    p.add_argument("--loss", default="arcface", choices=["arcface", "triplet"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Used only for arcface (regular sampling).")
    p.add_argument("--p", type=int, default=8, help="Identities per batch (triplet).")
    p.add_argument("--k", type=int, default=4, help="Samples per identity (triplet).")
    p.add_argument("--triplet-margin", type=float, default=0.30)
    p.add_argument("--arcface-s", type=float, default=30.0)
    p.add_argument("--arcface-m", type=float, default=0.30)
    p.add_argument("--gei-h", type=int, default=64)
    p.add_argument("--gei-w", type=int, default=44)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transform(h: int, w: int) -> transforms.Compose:
    """Aspect-aware GEI transform, paired with v2 preprocessing.

    Note: the GEIs on disk should already be aspect-correctly aligned by
    :func:`modules.gait.src.preprocessing.align_silhouette` (or the
    aspect-aware path in ``phase1_video_to_gei.py``). The Resize here is a
    safety net for older datasets and exact-shape enforcement.
    """
    return transforms.Compose([
        transforms.Resize((h, w)),
        # Light augmentation — silhouettes only — keeps inference distribution close.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def build_train_dataset(data_dir: str, h: int, w: int) -> CASIABDataset:
    train_subjects = [f"{i:03d}" for i in range(1, 75)]
    return CASIABDataset(
        data_dir=data_dir,
        subject_list=train_subjects,
        transform=build_transform(h, w),
    )


# ---------------------------------------------------------------------------
# Train loops
# ---------------------------------------------------------------------------
def _make_model(num_classes: int, device: torch.device) -> BaselineGaitCNN:
    model = BaselineGaitCNN(num_classes=num_classes)
    return model.to(device)


def train_arcface(args, device: torch.device) -> BaselineGaitCNN:
    dataset = build_train_dataset(args.data, args.gei_h, args.gei_w)
    num_classes = len(dataset.subject_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = _make_model(num_classes, device)
    head = ArcFaceHead(
        embedding_dim=512,
        num_classes=num_classes,
        s=args.arcface_s,
        m=args.arcface_m,
    ).to(device)

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    print(f"[arcface] {len(dataset)} samples, {num_classes} classes, "
          f"batch={args.batch_size}, epochs={args.epochs}, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            _, emb = model(images, return_embedding=True)
            logits = head(emb, labels)
            loss = ce(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        sched.step()
        print(f"epoch {epoch:>3}/{args.epochs}  loss={running / max(seen, 1):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.1f}s)")

    return model


def train_triplet(args, device: torch.device) -> BaselineGaitCNN:
    dataset = build_train_dataset(args.data, args.gei_h, args.gei_w)
    num_classes = len(dataset.subject_to_idx)

    sampler = PKBatchSampler(
        labels_from_dataset(dataset),
        p=args.p,
        k=args.k,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _make_model(num_classes, device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    triplet = BatchHardTripletLoss(margin=args.triplet_margin)

    print(f"[triplet] {len(dataset)} samples, {num_classes} classes, "
          f"P={args.p}×K={args.k}={args.p*args.k}/batch, epochs={args.epochs}, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            _, emb = model(images, return_embedding=True)
            # L2-normalize before loss; keeps geometry consistent with inference.
            emb_n = F.normalize(emb, p=2, dim=1)
            loss = triplet(emb_n, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item()
            n += 1

        sched.step()
        print(f"epoch {epoch:>3}/{args.epochs}  loss={running / max(n, 1):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.1f}s)")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )

    if args.loss == "arcface":
        model = train_arcface(args, device)
    elif args.loss == "triplet":
        model = train_triplet(args, device)
    else:  # pragma: no cover - argparse choices guard this.
        raise ValueError(f"Unknown loss: {args.loss}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[saved] {out_path}")

    # Write the sidecar config so inference picks up the v2 pipeline.
    config = GaitConfig(
        model_version=f"v2-resnet18-{args.loss}",
        preproc_version=f"v2-aspect{args.gei_h}x{args.gei_w}-cycle",
        arch=args.arch,
        num_classes=GaitConfig().num_classes,  # Inference always uses 74-class head shape;
                                               # the head is dropped in BaselineGaitCNN.fc=Identity.
        embedding_dim=512,
        gei_size=(args.gei_h, args.gei_w),
        l2_normalize_embeddings=True,
        score_scaling="raw_cosine",
        notes=f"trained with --loss {args.loss}",
    )
    sidecar = sidecar_path_for(out_path)
    config.save(sidecar)
    print(f"[saved] {sidecar}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
