"""
Benchmark/verification for the YOLO model-load caching in gait_utils.

We don't depend on torch / ultralytics being installed — instead, we patch
the lazy-loader hooks to count how many times the YOLO model is constructed
under the OLD behaviour vs. the NEW (cached) behaviour, and how many times
the dispatch in `video_to_gei` falls back to MOG2 when YOLO succeeds.

This is the meaningful "proof" for two of the bugfixes/improvements:

  1. (correctness) `video_to_gei` previously called `_yolo_video_to_gei` and
     **discarded its return value**, then fell through to MOG2 every time.
     The new code returns the YOLO result and only falls back on failure.

  2. (perf) `_yolo_video_to_gei` previously instantiated `YOLO(...)` on every
     call. The new code caches the model singleton via `_get_yolo()`.

Run:
    python modules/gait/src/benchmarks/bench_yolo_dispatch.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)


# --- Stub torch / torchvision / PIL.Image / cv2 / BaselineGaitCNN  ---------
# So we can import gait_utils without the heavy ML stack installed.
def _install_stubs():
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_mod.device = lambda x: x
        torch_mod.no_grad = lambda: _DummyCtx()
        torch_mod.load = lambda *a, **k: {}
        sys.modules["torch"] = torch_mod

    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv_transforms = types.ModuleType("torchvision.transforms")
        tv_transforms.Compose = lambda x: (lambda img: img)
        tv_transforms.Resize = lambda s: None
        tv_transforms.ToTensor = lambda: None
        tv_transforms.Normalize = lambda **kw: None
        tv.transforms = tv_transforms
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = tv_transforms

    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.fromarray = lambda x: x
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image
    # Stub the deep model module so gait_utils import doesn't pull torch.nn etc.
    pkg_modules = "modules.gait.src.phase3_dataset_and_model"
    if pkg_modules not in sys.modules:
        m = types.ModuleType(pkg_modules)
        class BaselineGaitCNN:
            def __init__(self, *a, **k): pass
            def to(self, *a, **k): return self
            def eval(self): return self
            def load_state_dict(self, *a, **k): pass
        m.BaselineGaitCNN = BaselineGaitCNN
        sys.modules[pkg_modules] = m


class _DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_install_stubs()
import numpy as np  # after stubs

import gait_utils  # noqa: E402


# --- Counters --------------------------------------------------------------
yolo_construct_calls = 0
yolo_inference_calls = 0
fallback_calls = 0


class FakeYOLO:
    def __init__(self, weights):
        global yolo_construct_calls
        yolo_construct_calls += 1

    def __call__(self, frame, **kwargs):
        global yolo_inference_calls
        yolo_inference_calls += 1
        # Return a "result" with a usable mask + box.
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.float32) * 0.5
        boxes_xyxy = np.array([[0, 0, w, h]], dtype=np.float32)

        class _Tensor:
            def __init__(self, arr):
                self.arr = arr
            def cpu(self): return self
            def numpy(self): return self.arr
        masks_obj = types.SimpleNamespace(data=[_Tensor(mask)])
        boxes_obj = types.SimpleNamespace(xyxy=_Tensor(boxes_xyxy))
        result = types.SimpleNamespace(masks=masks_obj, boxes=boxes_obj)
        return [result]


# --- Patch ultralytics import path -----------------------------------------
ultra = types.ModuleType("ultralytics")
ultra.YOLO = FakeYOLO
sys.modules["ultralytics"] = ultra


# --- Patch cv2 minimally to drive _yolo_video_to_gei -----------------------
class _FakeCap:
    """Yields N synthetic frames then EOF."""
    def __init__(self, n=10):
        self.n = n
        self.i = 0
    def isOpened(self): return True
    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.zeros((120, 160, 3), dtype=np.uint8)
    def release(self): pass


import cv2  # the stub
cv2.VideoCapture = lambda path: _FakeCap(n=10)
cv2.resize = lambda img, size: np.zeros(size[::-1], dtype=np.uint8) \
    if hasattr(img, "shape") else img
cv2.cvtColor = lambda img, code: img
cv2.COLOR_BGR2GRAY = 0
cv2.createBackgroundSubtractorMOG2 = lambda **kw: None
cv2.MORPH_ELLIPSE = 0
cv2.MORPH_CLOSE = 0
cv2.MORPH_OPEN = 0
cv2.RETR_EXTERNAL = 0
cv2.CHAIN_APPROX_SIMPLE = 0


# Patch the fallback so we can count when it gets invoked.
_orig_fallback = gait_utils._fallback_video_to_gei
def _counting_fallback(path):
    global fallback_calls
    fallback_calls += 1
    return np.zeros((64, 64), dtype=np.uint8)
gait_utils._fallback_video_to_gei = _counting_fallback


# Reset cache so first call exercises the loader.
gait_utils._yolo_model = None


# --- Run NEW behaviour: many video calls -----------------------------------
N_VIDEOS = 25
for _ in range(N_VIDEOS):
    out = gait_utils.video_to_gei("dummy.mp4")
    assert out.shape == (64, 64), out.shape

new_construct = yolo_construct_calls
new_fallback  = fallback_calls

# --- Simulate OLD behaviour for the same N videos --------------------------
# OLD: a) constructed YOLO inside _yolo_video_to_gei every call,
#      b) discarded its result, c) ALWAYS fell through to MOG2.
yolo_construct_calls = 0
fallback_calls = 0
for _ in range(N_VIDEOS):
    # (a) one YOLO construction per video
    _ = FakeYOLO("yolov8n-seg.pt")
    # (b) result discarded
    # (c) fallback always runs
    fallback_calls += 1

old_construct = yolo_construct_calls
old_fallback  = fallback_calls


# --- Report ---------------------------------------------------------------
print(f"Videos processed: {N_VIDEOS}")
print()
print(f"{'Metric':<35} {'OLD':>10} {'NEW':>10}")
print("-" * 57)
print(f"{'YOLO model constructions':<35} {old_construct:>10} {new_construct:>10}")
print(f"{'MOG2 fallback invocations':<35} {old_fallback:>10} {new_fallback:>10}")
print()
print(f"YOLO load reduction: {old_construct} → {new_construct} "
      f"({old_construct - new_construct} avoided)")
print(f"Correctness fix: fallback used {old_fallback}/{N_VIDEOS} times in OLD "
      f"(always), {new_fallback}/{N_VIDEOS} times in NEW (only on YOLO failure)")
assert new_construct == 1, "YOLO must be constructed exactly once across calls"
assert new_fallback  == 0, "Fallback must NOT run when YOLO succeeds"
print("\n✅ Both invariants verified.")
