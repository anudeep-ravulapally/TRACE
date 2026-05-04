"""
GaitConfig — model + preprocessing version bundle.

Why this exists:
    The gait pipeline has two pieces that MUST move together: the trained model
    weights (.pth) and the preprocessing that produced the inputs the model was
    trained on (GEI size, normalization, whether embeddings are L2-normalized,
    matching threshold, etc.). Embeddings are only comparable within the same
    (model, preproc) pair. Mix versions and accuracy silently collapses.

How it's used:
    - When training a new model, write a sidecar JSON next to the ``.pth`` with
      :meth:`GaitConfig.save`.
    - At inference, :func:`load_gait_config` looks for that sidecar JSON and
      falls back to a hardcoded legacy v1 config if missing — so the existing
      ``baseline_gait_model.pth`` (trained before this PR) keeps working.

Versioning rule of thumb:
    Bump ``model_version`` whenever the loss / classifier head changes.
    Bump ``preproc_version`` whenever GEI extraction, alignment, or
    normalization changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Legacy v1 defaults — what the original baseline_gait_model.pth was trained
# with. Keep these frozen; they are the fallback for any model file that
# doesn't ship a sidecar config.
# ---------------------------------------------------------------------------
LEGACY_V1 = {
    "model_version": "v1-resnet18-ce",
    "preproc_version": "v1-square64",
    "arch": "resnet18",
    "num_classes": 74,
    "embedding_dim": 512,
    "gei_size": [64, 64],          # H, W
    "normalize_mean": [0.5],
    "normalize_std": [0.5],
    "l2_normalize_embeddings": False,
    # v1 inference uses the Confidence Punisher hack — see gait_utils.scale_gait_score
    "score_scaling": "minmax_punisher",
    "minmax_base_min": 0.982,
    "minmax_base_max": 1.000,
    "match_threshold": 0.50,
    # Open-set unknown-person rejection (see gait_utils.find_best_gait_match).
    # v1's ResNet-18 + plain CE produces a heavily collapsed cosine band
    # (~0.978..0.989), so an unknown probe's top-1 against a multi-user gallery
    # can sneak past match_threshold. Two extra guardrails keep that out:
    #   * unknown_raw_floor — min absolute raw cosine for any accept.
    #   * unknown_margin_min — min gap between best-user and 2nd-best-user
    #     raw cosine. Genuine probes have one clear winner; unknowns score
    #     ~equally against everyone.
    "unknown_raw_floor": 0.987,
    "unknown_margin_min": 0.003,
    # Preprocessing knobs introduced in v2 — irrelevant for v1.
    "aspect_aware_align": False,
    "use_period_detection": False,
    "min_silhouette_frames": 5,
}


@dataclass
class GaitConfig:
    """Bundle of (model + preprocessing) parameters tied to a checkpoint.

    Defaults below describe the *new* v2 pipeline (aspect-aware preprocessing,
    L2-normalized ArcFace/triplet embeddings, raw-cosine threshold). Use
    :meth:`legacy_v1` to get the bundle that matches the existing checkpoint
    in this repo.
    """

    # ---- identity ---------------------------------------------------------
    model_version: str = "v2-resnet18-arcface"
    preproc_version: str = "v2-aspect64x44-cycle"

    # ---- model ------------------------------------------------------------
    arch: str = "resnet18"
    num_classes: int = 74
    embedding_dim: int = 512

    # ---- preprocessing ----------------------------------------------------
    gei_size: Tuple[int, int] = (64, 44)              # (H, W) — aspect-aware
    normalize_mean: Tuple[float, ...] = (0.5,)
    normalize_std: Tuple[float, ...] = (0.5,)
    aspect_aware_align: bool = True
    use_period_detection: bool = True
    min_silhouette_frames: int = 5

    # ---- inference --------------------------------------------------------
    l2_normalize_embeddings: bool = True
    # 'raw_cosine'  → use cosine similarity directly (production mode for v2).
    # 'minmax_punisher' → legacy v1 hack (see gait_utils.scale_gait_score).
    score_scaling: str = "raw_cosine"
    minmax_base_min: float = 0.982
    minmax_base_max: float = 1.000
    # Threshold operates on whatever score_scaling produces — should be set
    # from the verification ROC curve (see evaluate.py).
    match_threshold: float = 0.50

    # ---- open-set unknown rejection --------------------------------------
    # Minimum raw cosine for any accept. On v2 (wide cosine band) the absolute
    # threshold is enough, so this defaults to 0.0 (off). On v1 (collapsed
    # band) it's ~0.987 — see ``legacy_v1``.
    unknown_raw_floor: float = 0.0
    # Minimum gap between best-user and second-best-user raw cosine. Genuine
    # probes have one clear winner; unknowns score ~equally against everyone.
    # v2's wider band tolerates a looser default; v1 tightens this to ~0.003.
    unknown_margin_min: float = 0.05

    # ---- bookkeeping ------------------------------------------------------
    notes: str = ""
    extra: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- IO --
    @classmethod
    def legacy_v1(cls) -> "GaitConfig":
        """Return the config that matches the pre-PR baseline_gait_model.pth."""
        # Translate dict → dataclass, accepting the slightly different shapes.
        d = dict(LEGACY_V1)
        d["gei_size"] = tuple(d["gei_size"])
        d["normalize_mean"] = tuple(d["normalize_mean"])
        d["normalize_std"] = tuple(d["normalize_std"])
        return cls(**d)

    def save(self, path: str) -> None:
        """Write this config as a JSON sidecar (creates parent dirs)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = asdict(self)
        # JSON has no tuples — store as lists; load() rehydrates.
        d["gei_size"] = list(self.gei_size)
        d["normalize_mean"] = list(self.normalize_mean)
        d["normalize_std"] = list(self.normalize_std)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "GaitConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["gei_size"] = tuple(d.get("gei_size", (64, 44)))
        d["normalize_mean"] = tuple(d.get("normalize_mean", (0.5,)))
        d["normalize_std"] = tuple(d.get("normalize_std", (0.5,)))
        # Drop any unknown fields so we stay forward-compatible.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def sidecar_path_for(model_path: str) -> str:
    """Convention: ``foo.pth`` → ``foo.config.json`` next to it."""
    base, _ = os.path.splitext(model_path)
    return base + ".config.json"


def load_gait_config(model_path: Optional[str]) -> GaitConfig:
    """Load the sidecar config for a model file, falling back to legacy v1.

    This is the function inference code should call. It guarantees a config
    is always returned, so callers never need a "did this model ship a
    config?" branch.
    """
    if model_path:
        sidecar = sidecar_path_for(model_path)
        if os.path.exists(sidecar):
            try:
                return GaitConfig.load(sidecar)
            except Exception:
                # Corrupt sidecar — better to fall back than to crash inference.
                pass
    return GaitConfig.legacy_v1()
