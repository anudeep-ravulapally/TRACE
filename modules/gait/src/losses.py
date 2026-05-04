"""
Metric-learning losses for the gait module — Step 2 of the accuracy plan.

The original ``BaselineGaitCNN`` is trained with plain cross-entropy over 74
closed-set CASIA-B subjects, then matched at inference via cosine similarity
against new (unseen) users. That mismatch — softmax doesn't optimize for
cosine separability — is what the ``BASE_MIN=0.982`` "Confidence Punisher"
in ``gait_utils.py`` is papering over.

This module gives a training script two clean alternatives:

- :class:`ArcFaceHead` — additive angular margin softmax. Drop-in replacement
  for the final ``nn.Linear`` classifier; needs L2-normalized embeddings and
  produces well-separated cosine spaces for open-set retrieval.
- :class:`BatchHardTripletLoss` — pure metric loss for use with a
  P×K BatchSampler (see ``sampler.py``). No classifier head needed.

Both pair naturally with L2-normalized embeddings, so the production
inference path can just use raw cosine similarity (no Min-Max hack required).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    """Additive angular margin softmax (Deng et al., 2019).

    Replaces the final ``nn.Linear`` of a recognition model. Embeddings are
    L2-normalized inside the module, weights are L2-normalized too, and a
    margin ``m`` is added to the angle of the *target* class only.
    The result is multiplied by a scale ``s`` before cross-entropy, so the
    softmax temperature stays sane when cosines live in [-1, 1].

    Usage::

        head = ArcFaceHead(embedding_dim=512, num_classes=74)
        loss_fn = nn.CrossEntropyLoss()
        ...
        emb = backbone(x)               # (B, 512), unnormalized
        logits = head(emb, labels)      # (B, num_classes), training-mode
        loss = loss_fn(logits, labels)

    At inference, call ``head.cosine_logits(emb)`` to get plain cosine logits
    (no margin) — but typically you don't need the head at all at inference,
    you just L2-normalize the backbone output and do nearest-neighbour cosine.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.30,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trigonometric constants for the margin step.
        self._cos_m = math.cos(self.m)
        self._sin_m = math.sin(self.m)
        # Threshold beyond which we'd drop into the wrong-side region —
        # used to keep gradients stable for cos(theta + m).
        self._th = math.cos(math.pi - self.m)
        self._mm = math.sin(math.pi - self.m) * self.m

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Plain cosine logits — no margin, no scale. Useful for evaluation."""
        emb_n = F.normalize(embeddings, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        return emb_n @ w_n.t()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self.cosine_logits(embeddings).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        cos_phi = cosine * self._cos_m - sine * self._sin_m

        if self.easy_margin:
            cos_phi = torch.where(cosine > 0, cos_phi, cosine)
        else:
            # When cos(theta) < cos(pi - m), reverting keeps the loss monotonic.
            cos_phi = torch.where(cosine > self._th, cos_phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * cos_phi + (1.0 - one_hot) * cosine
        return logits * self.s


# ---------------------------------------------------------------------------
# Batch-hard triplet loss
# ---------------------------------------------------------------------------
class BatchHardTripletLoss(nn.Module):
    """Triplet loss with batch-hard mining (Hermans et al., 2017).

    For each anchor in a batch, the *hardest positive* (same identity, max
    distance) and *hardest negative* (different identity, min distance) are
    used. Requires batches sampled with a P×K strategy so each identity
    contributes K samples — see :class:`PKBatchSampler`.

    Inputs to ``forward`` are expected to already be L2-normalized; cosine
    distance is then ``1 - cos(a, b)``.
    """

    def __init__(self, margin: float = 0.30) -> None:
        super().__init__()
        self.margin = float(margin)

    @staticmethod
    def _pairwise_cosine_distances(emb: torch.Tensor) -> torch.Tensor:
        # emb is L2-normalized → cosine = emb @ emb.T → distance = 1 - cosine.
        sim = emb @ emb.t()
        # Clamp for numerical safety; allow tiny negatives that arise from fp.
        return (1.0 - sim).clamp(min=0.0)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings, p=2, dim=1)
        dist = self._pairwise_cosine_distances(emb)

        labels = labels.view(-1)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)         # (B, B) bool
        diff = ~same
        # Exclude self-pairs from the positive set.
        eye = torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)
        pos_mask = same & ~eye

        # Hardest positive: max distance among same-identity, non-self pairs.
        # Set non-positive entries to -inf so they don't win the max.
        neg_inf = torch.tensor(float("-inf"), device=emb.device)
        d_pos = torch.where(pos_mask, dist, neg_inf).max(dim=1).values

        # Hardest negative: min distance among different-identity pairs.
        # Set non-negative entries to +inf so they don't win the min.
        pos_inf = torch.tensor(float("inf"), device=emb.device)
        d_neg = torch.where(diff, dist, pos_inf).min(dim=1).values

        # Anchors with no valid positive (lone identity in batch) or no
        # valid negative (single-identity batch) are ignored.
        valid = torch.isfinite(d_pos) & torch.isfinite(d_neg)
        if not valid.any():
            return torch.zeros((), device=emb.device, requires_grad=True)

        losses = F.relu(d_pos[valid] - d_neg[valid] + self.margin)
        return losses.mean()
