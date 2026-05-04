"""
P×K BatchSampler for metric-learning gait training.

Triplet loss (and to a lesser extent ArcFace) wants every minibatch to contain
multiple identities, each with multiple samples, so that
``BatchHardTripletLoss`` has both intra-class and inter-class pairs to mine
within the batch.

This sampler emits batches of size ``P * K`` where:
  - P identities are sampled (without replacement, until the epoch resets)
  - K samples per identity are sampled (with replacement if needed)

Drop-in for ``DataLoader(batch_sampler=PKBatchSampler(...))``. The dataset
must expose integer labels — :class:`~modules.gait.src.phase3_dataset_and_model.CASIABDataset`
does (its ``samples`` is a list of ``(path, label)`` tuples).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List, Sequence, Tuple


class PKBatchSampler:
    """Yields batches of indices: P identities × K samples each."""

    def __init__(
        self,
        labels: Sequence[int],
        p: int = 8,
        k: int = 4,
        num_batches: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if p <= 0 or k <= 0:
            raise ValueError("p and k must be positive")
        self.p = int(p)
        self.k = int(k)
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)

        # Group sample indices by label.
        groups: dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            groups[int(lbl)].append(idx)
        # Drop identities that have no samples (defensive).
        self._groups: dict[int, List[int]] = {
            lbl: idxs for lbl, idxs in groups.items() if idxs
        }
        self._labels: List[int] = list(self._groups.keys())

        if len(self._labels) < self.p:
            raise ValueError(
                f"Need at least p={self.p} distinct identities, got {len(self._labels)}"
            )

        # Default epoch length: enough batches to cover the dataset roughly once.
        if num_batches is None:
            total_samples = sum(len(v) for v in self._groups.values())
            num_batches = max(1, total_samples // (self.p * self.k))
        self._num_batches = int(num_batches)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        labels_pool: List[int] = []
        for _ in range(self._num_batches):
            # Refill the identity pool when we run out (epoch-style cycle).
            if len(labels_pool) < self.p:
                labels_pool = list(self._labels)
                if self.shuffle:
                    self._rng.shuffle(labels_pool)
            chosen_labels = labels_pool[: self.p]
            labels_pool = labels_pool[self.p:]

            batch: List[int] = []
            for lbl in chosen_labels:
                idxs = self._groups[lbl]
                if len(idxs) >= self.k:
                    batch.extend(self._rng.sample(idxs, self.k))
                else:
                    # Sample with replacement when an identity is short on samples.
                    batch.extend(self._rng.choices(idxs, k=self.k))
            yield batch


def labels_from_dataset(dataset) -> List[int]:
    """Extract the per-sample integer label list from a CASIABDataset.

    Kept as a tiny helper so callers don't reach into ``dataset.samples``.
    """
    return [int(lbl) for _, lbl in dataset.samples]
