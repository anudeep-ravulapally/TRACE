"""Unit tests for gait_utils.find_best_gait_match open-set unknown-rejection.

These tests stub out the User model with a tiny dataclass so we can
hand-craft galleries and inject probes whose top-1/top-2 cosine geometry
exercises each rejection branch.

Run with: ``pytest modules/gait/tests/test_unknown_rejection.py``
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

# Make the repo root importable when pytest is invoked from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gait_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures — fake User and embedding helpers
# ---------------------------------------------------------------------------
@dataclass
class FakeUser:
    name: str
    clips: List[List[float]] = field(default_factory=list)

    def get_gait_embeddings(self):
        return list(self.clips)

    def get_gait_embedding(self):
        return self.clips[0] if self.clips else None


def _vec_with_cosine(reference: np.ndarray, target_cos: float, seed: int) -> np.ndarray:
    """Construct a unit vector whose cosine with ``reference`` is ``target_cos``.

    Uses the standard recipe: pick a random direction, project it onto the
    plane orthogonal to ``reference``, then mix with ``reference`` at the
    requested angle.
    """
    rng = np.random.default_rng(seed)
    ref = reference / np.linalg.norm(reference)
    rand = rng.standard_normal(ref.shape).astype(np.float32)
    perp = rand - np.dot(rand, ref) * ref
    perp /= (np.linalg.norm(perp) + 1e-12)
    target_cos = float(np.clip(target_cos, -1.0, 1.0))
    out = target_cos * ref + np.sqrt(max(0.0, 1.0 - target_cos * target_cos)) * perp
    return out.astype(np.float32)


@pytest.fixture
def reset_constants():
    """Snapshot/restore the open-set constants so each test is independent."""
    saved = (
        gait_utils.GAIT_THRESHOLD,
        gait_utils.UNKNOWN_RAW_FLOOR,
        gait_utils.UNKNOWN_MARGIN_MIN,
        gait_utils._USE_RAW_COSINE,
        gait_utils.BASE_MIN,
        gait_utils.BASE_MAX,
    )
    yield
    (
        gait_utils.GAIT_THRESHOLD,
        gait_utils.UNKNOWN_RAW_FLOOR,
        gait_utils.UNKNOWN_MARGIN_MIN,
        gait_utils._USE_RAW_COSINE,
        gait_utils.BASE_MIN,
        gait_utils.BASE_MAX,
    ) = saved


def _configure_v1_legacy(monkeypatch):
    """Force the v1 (Min-Max Punisher) score path with calibrated knobs."""
    monkeypatch.setattr(gait_utils, "_USE_RAW_COSINE", False)
    monkeypatch.setattr(gait_utils, "BASE_MIN", 0.982)
    monkeypatch.setattr(gait_utils, "BASE_MAX", 1.000)
    monkeypatch.setattr(gait_utils, "GAIT_THRESHOLD", 0.50)
    monkeypatch.setattr(gait_utils, "UNKNOWN_RAW_FLOOR", 0.987)
    monkeypatch.setattr(gait_utils, "UNKNOWN_MARGIN_MIN", 0.003)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_genuine_probe_accepts(monkeypatch, reset_constants):
    """A probe close to one user, well separated from the rest, should match."""
    _configure_v1_legacy(monkeypatch)

    rng = np.random.default_rng(0)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Three users in the v1 collapsed band, with the genuine one slightly higher.
    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.995, seed=1).tolist()])
    u2 = FakeUser("bob",   [_vec_with_cosine(base, 0.984, seed=2).tolist()])
    u3 = FakeUser("carol", [_vec_with_cosine(base, 0.985, seed=3).tolist()])
    probe = u1.clips[0]  # exact match → cosine 1.0 with alice

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1, u2, u3])

    assert user is u1
    assert reason is None
    assert raw == pytest.approx(1.0, abs=1e-5)
    assert scaled >= gait_utils.GAIT_THRESHOLD


def test_unknown_probe_below_raw_floor(monkeypatch, reset_constants):
    """Probe whose top-1 raw cosine is under the floor → ``below_raw_floor``."""
    _configure_v1_legacy(monkeypatch)

    rng = np.random.default_rng(10)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Build a single-user gallery where the unknown probe's similarity is
    # ABOVE the punisher threshold (so step 1 passes) but BELOW the raw floor.
    # Punisher: scaled = (raw - 0.982) / 0.018  → raw = 0.9855 yields scaled = 0.305? no.
    # Need scaled >= 0.50 → raw >= 0.982 + 0.5*0.018 = 0.991.
    # So pick raw = 0.9925 → scaled ≈ 0.583 (passes threshold),
    # and floor = 0.987 actually accepts that. We need a multi-user gallery
    # where best_raw < 0.987 but the punisher threshold is also breached.
    # Easier: lower the threshold for this test so the floor is the binding
    # constraint, mirroring the real-world v1 collapsed-band behavior.
    monkeypatch.setattr(gait_utils, "GAIT_THRESHOLD", 0.0)

    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.984, seed=11).tolist()])
    u2 = FakeUser("bob",   [_vec_with_cosine(base, 0.983, seed=12).tolist()])
    probe = base.tolist()  # cosine ~0.984 with both

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1, u2])

    assert user is None
    assert reason == "below_raw_floor"
    assert raw < gait_utils.UNKNOWN_RAW_FLOOR


def test_unknown_probe_ambiguous_margin(monkeypatch, reset_constants):
    """top-1 ≥ raw floor but top1−top2 < margin_min → ``ambiguous_match``."""
    _configure_v1_legacy(monkeypatch)
    # Lower threshold so the punisher-scaled score doesn't reject first.
    monkeypatch.setattr(gait_utils, "GAIT_THRESHOLD", 0.0)

    rng = np.random.default_rng(20)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Two users with nearly identical similarity to the probe — classic
    # unknown fingerprint. Both well above the raw floor.
    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.9925, seed=21).tolist()])
    u2 = FakeUser("bob",   [_vec_with_cosine(base, 0.9924, seed=22).tolist()])
    u3 = FakeUser("carol", [_vec_with_cosine(base, 0.9923, seed=23).tolist()])
    probe = base.tolist()

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1, u2, u3])

    assert user is None
    assert reason == "ambiguous_match"
    assert raw >= gait_utils.UNKNOWN_RAW_FLOOR


def test_unknown_probe_below_threshold(monkeypatch, reset_constants):
    """When the scaled score itself is under threshold, that wins as the reason."""
    _configure_v1_legacy(monkeypatch)

    rng = np.random.default_rng(30)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # All users far below the punisher's MinMax base → scaled = 0.
    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.50, seed=31).tolist()])
    u2 = FakeUser("bob",   [_vec_with_cosine(base, 0.45, seed=32).tolist()])
    probe = base.tolist()

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1, u2])

    assert user is None
    assert reason == "below_threshold"
    assert scaled < gait_utils.GAIT_THRESHOLD


def test_single_user_gallery_skips_margin(monkeypatch, reset_constants):
    """One enrolled user → margin check is bypassed; only threshold + floor apply."""
    _configure_v1_legacy(monkeypatch)

    rng = np.random.default_rng(40)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Genuine match; would be flagged ``ambiguous_match`` if a fictitious
    # second user existed nearby, but with a single-user gallery there is no
    # second-best to compare against.
    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.998, seed=41).tolist()])
    probe = u1.clips[0]

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1])

    assert user is u1
    assert reason is None


def test_empty_gallery_returns_below_threshold(monkeypatch, reset_constants):
    _configure_v1_legacy(monkeypatch)
    user, scaled, raw, reason = gait_utils.find_best_gait_match([0.1] * 64, [])
    assert user is None
    assert reason == "below_threshold"


def test_v2_raw_cosine_path_uses_loose_margin(monkeypatch, reset_constants):
    """v2 (raw_cosine, wider band) defaults — sanity check no regression."""
    monkeypatch.setattr(gait_utils, "_USE_RAW_COSINE", True)
    monkeypatch.setattr(gait_utils, "GAIT_THRESHOLD", 0.50)
    monkeypatch.setattr(gait_utils, "UNKNOWN_RAW_FLOOR", 0.0)        # off
    monkeypatch.setattr(gait_utils, "UNKNOWN_MARGIN_MIN", 0.05)

    rng = np.random.default_rng(50)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)

    u1 = FakeUser("alice", [_vec_with_cosine(base, 0.90, seed=51).tolist()])
    u2 = FakeUser("bob",   [_vec_with_cosine(base, 0.40, seed=52).tolist()])
    probe = u1.clips[0]

    user, scaled, raw, reason = gait_utils.find_best_gait_match(probe, [u1, u2])
    assert user is u1
    assert reason is None
