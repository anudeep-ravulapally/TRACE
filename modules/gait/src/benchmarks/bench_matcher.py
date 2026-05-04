"""
Benchmark: vectorized vs. per-user gait matcher.

Compares the OLD per-user-loop implementation of `find_best_gait_match`
(which calls sklearn.cosine_similarity once per stored user) against the
NEW vectorized implementation (single matmul over a stacked matrix).

Both implementations are run on the SAME synthetic gallery and the SAME
query embeddings, and we assert that:
  - the picked user is the same
  - the raw and scaled scores match (within float tolerance)

Then we report wall-clock timings.

Run:
    python modules/gait/src/benchmarks/bench_matcher.py
"""
import os
import sys
import time
import numpy as np

# Make repo root importable
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Re-implement the OLD matcher locally so we can compare without git history.
# This is byte-for-byte the previous implementation in gait_utils.py.
# ---------------------------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

GAIT_THRESHOLD = 0.50
BASE_MIN = 0.982
BASE_MAX = 1.000


def _scale(raw_score: float) -> float:
    if raw_score < BASE_MIN:
        return 0.0
    scaled = (raw_score - BASE_MIN) / (BASE_MAX - BASE_MIN)
    return float(min(max(scaled, 0.0), 1.0))


def find_best_gait_match_OLD(new_embedding, all_users):
    best_user = None
    best_raw = -1.0
    best_scaled = 0.0
    new_vec = np.array(new_embedding).reshape(1, -1)

    for user in all_users:
        stored = user.get_gait_embedding()
        if stored is None:
            continue
        stored_vec = np.array(stored).reshape(1, -1)
        raw = float(sk_cosine(new_vec, stored_vec)[0][0])
        scaled = _scale(raw)
        if scaled > best_scaled:
            best_raw = raw
            best_scaled = scaled
            best_user = user
        elif scaled == best_scaled and raw > best_raw:
            best_raw = raw
            best_user = user

    if best_scaled >= GAIT_THRESHOLD:
        return best_user, best_scaled, best_raw
    return None, best_scaled, best_raw


# ---------------------------------------------------------------------------
# NEW matcher: pulled out of gait_utils.py without importing torch/torchvision
# (which the matcher itself does not need, and which aren't always installed
# in lightweight benchmark environments).
# ---------------------------------------------------------------------------
def _load_new_matcher():
    src_path = os.path.join(REPO_ROOT, "gait_utils.py")
    with open(src_path) as f:
        src = f.read()
    ns = {"np": np, "BASE_MIN": BASE_MIN, "BASE_MAX": BASE_MAX,
          "GAIT_THRESHOLD": GAIT_THRESHOLD,
          # The new matcher reads two module-level config flags. Inject the
          # values that match the legacy v1 model the benchmark targets.
          "_USE_RAW_COSINE": False, "_L2_NORMALIZE": False,
          # Open-set knobs introduced for unknown-person rejection. Set to 0
          # here so the benchmark exercises the same accept/reject decisions
          # as the OLD matcher (no extra rejection branches).
          "UNKNOWN_RAW_FLOOR": 0.0, "UNKNOWN_MARGIN_MIN": 0.0}
    # Pull the matcher *and* its private clip-embedding helper. The helper is
    # defined just before find_best_gait_match in gait_utils.py.
    start_helper = src.index("def _user_clip_embeddings(")
    fn_src = src[start_helper:]
    exec(fn_src, ns)
    return ns["find_best_gait_match"]


# ---------------------------------------------------------------------------
# Synthetic User stub matching the .get_gait_embedding() shape.
# ---------------------------------------------------------------------------
class _StubUser:
    __slots__ = ("name", "_emb")

    def __init__(self, name, emb):
        self.name = name
        self._emb = emb

    def get_gait_embedding(self):
        return self._emb


def _make_gallery(n_users, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    users = []
    for i in range(n_users):
        v = rng.standard_normal(dim).astype(np.float32)
        users.append(_StubUser(f"u{i:04d}", v.tolist()))
    return users


def _make_query_close_to(user, jitter=0.02, seed=1):
    rng = np.random.default_rng(seed)
    base = np.asarray(user.get_gait_embedding(), dtype=np.float32)
    noise = rng.standard_normal(base.shape).astype(np.float32) * jitter
    return (base + noise).tolist()


def _bench(fn, query, users, repeats):
    fn(query, users)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn(query, users)
    return time.perf_counter() - t0, result


def main():
    new_matcher = _load_new_matcher()

    print(f"{'Gallery size':>14} | {'OLD (s)':>10} | {'NEW (s)':>10} | "
          f"{'Speedup':>9} | {'Equiv?':>8}")
    print("-" * 62)

    sizes = [50, 200, 1000, 5000]
    repeats_for_size = {50: 2000, 200: 1000, 1000: 200, 5000: 50}

    for n in sizes:
        users = _make_gallery(n)
        target = users[n // 3]
        query = _make_query_close_to(target)
        repeats = repeats_for_size[n]

        t_old, r_old = _bench(find_best_gait_match_OLD, query, users, repeats)
        t_new, r_new = _bench(new_matcher,             query, users, repeats)

        same_user = (r_old[0] is r_new[0]) or (
            r_old[0] is not None and r_new[0] is not None
            and r_old[0].name == r_new[0].name
        )
        scaled_close = abs(r_old[1] - r_new[1]) < 1e-4
        raw_close    = abs(r_old[2] - r_new[2]) < 1e-4
        ok = "OK" if (same_user and scaled_close and raw_close) else "MISMATCH"

        speedup = t_old / t_new if t_new > 0 else float("inf")
        print(f"{n:>14} | {t_old:>10.4f} | {t_new:>10.4f} | "
              f"{speedup:>8.1f}x | {ok:>8}")


if __name__ == "__main__":
    main()
