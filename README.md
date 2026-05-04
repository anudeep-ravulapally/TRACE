# TRACE — Time & Recognition Automated Check-in Engine

> **Multimodal Biometric Attendance System**
> Face Recognition · Gait Recognition · Adaptive Score-Level Fusion

TRACE is a B.Tech Capstone research project demonstrating automated, contactless attendance using two biometric modalities fused together. Built with Flask, DeepFace, YOLOv8, and PyTorch, it ensures robust identification even when a user's face is partially occluded.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Open-Set Confidence Punisher](#open-set-confidence-punisher)
- [Improving Accuracy: Roadmap & Tools](#improving-accuracy-roadmap--tools)
- [Standard Operating Procedures](#standard-operating-procedures)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [How to Use TRACE](#how-to-use-trace)
- [Project Structure](#project-structure)
- [Privacy & Security](#privacy--security)
- [Authors](#authors)

---

## System Architecture

TRACE operates on a three-tier recognition pipeline:

| Module | Technology | Output |
|---|---|---|
| **Face Recognition** | DeepFace + ArcFace (512-D embedding) | `score_face` |
| **Gait Recognition** | YOLOv8-seg → GEI → ResNet-18 (512-D embedding) | `score_gait` |
| **Fusion Engine** | Adaptive score-level fusion | `score_final` |

### Adaptive Fusion Logic

The fusion engine dynamically weights both modalities based on a face visibility proxy score:

```
Face clearly visible  →  score_final = (0.7 × score_face) + (0.3 × score_gait)
Face occluded         →  score_final = (0.3 × score_face) + (0.7 × score_gait)
```

---

## Open-Set Confidence Punisher

The CASIA-B-trained ResNet-18 produces cosine similarity scores cramped in a narrow band (~0.982–1.0), which causes the model to force matches on imposters. A naive threshold fails because the margin between a true match and an imposter can be as small as **0.005**.

TRACE solves this with a strict **Min-Max Normalization scaler**:

| Subject | Raw Cosine Score | Scaled Confidence | Decision |
|---|---|---|---|
| True match (Anurag) | 0.99140 | **52.2%** → displays as ~85.6% | ✅ Accepted |
| Imposter (Hemanth) | 0.98624 | **23.5%** | ❌ Rejected |

**How it works:**

1. **Crush Zone** — Any raw score below `BASE_MIN = 0.982` is immediately set to `0.0` confidence. Imposters never reach the threshold.
2. **Stretch** — Scores between `0.982` and `1.0` are linearly scaled to `[0.0 → 1.0]`, turning a 0.005 raw margin into a **0.287 scaled margin**.
3. **Security Threshold** — The open-set gate checks scaled confidence ≥ `0.50`. The backend never relaxes this.
4. **Vanity Curve** — Passing scores `[0.50 → 1.0]` are re-mapped to `[85% → 99%]` for the UI only, so a borderline match displays as 85% rather than 50%.

> **Note:** the Confidence Punisher is a *symptom* of training the model with cross-entropy on a closed 74-class set and then matching unseen identities by cosine — softmax doesn't optimize for cosine separability, so unrelated embeddings collapse into a tiny band. The path forward is to retrain the backbone with a metric-learning loss (ArcFace / batch-hard triplet) so cosine scores naturally spread across `[0, 1]` and the punisher hack can be removed. The infrastructure for that retraining ships in this repo — see [Improving Accuracy](#improving-accuracy-roadmap-tools).

---

## Improving Accuracy: Roadmap & Tools

The gait module ships with everything needed to push accuracy from "demo-grade" to "production-grade" without rewriting the pipeline. The recommended order is:

| # | Step | What it gives you | Code |
|---|---|---|---|
| 0 | **Measure** | CASIA-B Rank-1 per condition (NM/BG/CL) + open-set verification ROC | `modules/gait/src/evaluate.py` |
| 1 | **Fix preprocessing** | Aspect-aware (64×44) centroid alignment, best-mask selection, silhouette QC, period-aware GEI | `modules/gait/src/preprocessing.py` |
| 2 | **Retrain with metric learning** | ArcFace head or batch-hard triplet on L2-normalized embeddings — eliminates the Confidence Punisher hack | `modules/gait/src/{losses,sampler,train}.py` |
| 3 | Architecture upgrade | Swap ResNet-18+GEI for GaitSet/GaitPart on silhouette sequences | *future work* |
| 4 | Hyperparameter tuning | Optuna over margin, embedding dim, P×K, LR schedule, augmentations | *future work* |
| 5 | **Test-time tricks** | Flip TTA, multi-clip enrollment, top-1 score fusion across clips, calibrated threshold | `gait_utils.py` (already active) |
| 6 | Domain adaptation | Fine-tune on a small in-domain dataset from the deployment camera | *future work* |
| 7 | **Production hygiene** | Versioned `(model.pth, preproc)` config bundle so embeddings stay comparable | `modules/gait/src/preproc_config.py` |

### Test-time improvements active out of the box

- **Multi-clip enrollment.** `POST /api/register-gait` now accepts a `videos` field (repeat the form name once per clip) and an optional `append=true` flag. The matcher reduces to the **maximum** cosine similarity across a user's clips. Single-clip enrollment via the `video` field still works unchanged.
- **Flip TTA.** `gei_to_embedding` averages the embedding of the GEI and its horizontal flip — silhouettes are roughly left-right symmetric so this is a free accuracy bump.
- **Vectorised gallery match.** Single matmul over the stacked gallery — see `modules/gait/src/benchmarks/bench_matcher.py` for the 10–15× speed-up.

### Versioning: model + preproc move together

`gait_utils` reads a sidecar JSON next to the `.pth` (e.g. `baseline_gait_model.config.json`). It captures `gei_size`, `score_scaling`, `l2_normalize_embeddings`, `match_threshold`, and the rest of the inference contract. **If the sidecar is missing, legacy v1 defaults are used** — the existing `baseline_gait_model.pth` keeps working byte-for-byte. Newly-trained models ship their own sidecar via `train.py`, automatically activating the v2 inference path (raw cosine, no `BASE_MIN` hack).

### Running it

```bash
# Step 0 — Evaluate the current model on CASIA-B GEIs.
python -m modules.gait.src.evaluate \
    --data ./dataset/GEI_Data \
    --model ./modules/gait/models/baseline_gait_model.pth \
    --per-angle

# Step 1 — Re-extract GEIs with the v2 aspect-aware pipeline.
python modules/gait/src/phase1_video_to_gei.py \
    --raw ./Raw_Video_Data --out ./dataset/TRACE_Gallery \
    --aspect-aware --gei-h 64 --gei-w 44 --detect-period

# Step 2 — Retrain with ArcFace.
python -m modules.gait.src.train \
    --data ./dataset/GEI_Data \
    --out  ./modules/gait/models/gait_v2.pth \
    --loss arcface --epochs 60 --gei-h 64 --gei-w 44

# Step 0 again — gate against the production targets:
#   NM Rank-1 ≥ 95%   BG Rank-1 ≥ 85%   CL Rank-1 ≥ 70%
#   TAR @ FAR=1e-3   ≥ 95%
python -m modules.gait.src.evaluate \
    --data ./dataset/GEI_Data --model ./modules/gait/models/gait_v2.pth
```

To switch the running app to the new model, replace the file at `modules/gait/models/baseline_gait_model.pth` (and ship its `.config.json` sidecar alongside).

---

## Standard Operating Procedures

The gait pipeline relies on accurate silhouette extraction (Gait Energy Images). Users **must** follow these rules during both registration and identification:

**1. 🎒 The Backpack Rule**
Remove heavy backpacks or oversized coats. YOLOv8 segments based on anatomical shape — external objects create artificial blobs that corrupt the GEI.

**2. 📐 The Camera Angle Rule**
Record from a **90° side profile** (walking left-to-right or right-to-left). Front-facing videos hide stride mechanics and produce near-identical GEIs across users.

**3. 📱 The Orientation Rule**
Both enrollment and inference videos **must be shot in Landscape mode**. Portrait mode distorts the aspect ratio during the 64×64 neural network resize, causing false negatives.

> When the v2 aspect-aware preprocessing pipeline is active (a sidecar config alongside the model declares `aspect_aware_align: true`), this rule relaxes — silhouettes are centroid-aligned and resized to a fixed aspect (default 64×44) without distortion. Landscape is still preferred but no longer mandatory.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | **3.9 – 3.11** | 3.12+ is **not supported** (TensorFlow constraint) |
| Git | Any | — |
| Webcam | — | Built-in or USB |

> ⚠️ **Python 3.12+ is not supported.** TensorFlow (used internally by DeepFace) requires Python 3.9–3.11.

---

## Setup & Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd TRACE_FINAL_PROJECT
```

**2. Create a virtual environment**

```bash
# Windows
python -m venv trace_venv
trace_venv\Scripts\activate

# macOS / Linux
python3 -m venv trace_venv
source trace_venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> This installs PyTorch, TensorFlow, DeepFace, Ultralytics (YOLOv8), and scikit-learn. A stable internet connection is required for the first install.

**4. Run the application**

```bash
python app.py
```

Visit → **http://127.0.0.1:5000**

---

## How to Use TRACE

### Step 1 — Register a Face

Navigate to **Register Face** (`/register`). Enter the user's name and department, then capture a 5–10 frame webcam sequence. The 512-D ArcFace embedding is averaged across frames and stored in the local SQLite database.

### Step 2 — Register Gait Biometrics

Navigate to **Register Gait** (`/register-gait`). Select the previously registered user from the dropdown and upload a **3–8 second walking video** (landscape orientation, side profile, no backpack).

### Step 3 — Take Attendance

| Mode | URL | What happens |
|---|---|---|
| **Face Only** | `/identify` | Live webcam snapshot matched via ArcFace cosine similarity |
| **Gait Only** | `/identify-gait` | Walking video → GEI extraction → ResNet-18 embedding → match |
| **Fusion** ⚡ | `/fusion` | Both modalities combined with adaptive weighting |

---

## Project Structure

```
TRACE_FINAL_PROJECT/
│
├── app.py                   # Flask application — API routing & fusion engine
├── face_utils.py            # ArcFace embedding extraction & cosine matching
├── gait_utils.py            # GEI pipeline, ResNet-18 inference, Confidence Punisher
├── models.py                # SQLAlchemy ORM models (User, embeddings)
├── requirements.txt         # Python dependencies
│
├── modules/
│   └── gait/
│       ├── src/
│       │   ├── phase1_video_to_gei.py     # GEI extraction (v1 legacy + v2 aspect-aware)
│       │   ├── phase3_dataset_and_model.py# CASIABDataset + BaselineGaitCNN
│       │   ├── preprocessing.py           # v2 silhouette alignment, QC, period-aware GEI
│       │   ├── losses.py                  # ArcFace + batch-hard triplet
│       │   ├── sampler.py                 # P×K BatchSampler for metric learning
│       │   ├── train.py                   # Metric-learning training entrypoint
│       │   ├── evaluate.py                # CASIA-B Rank-1/ROC harness
│       │   ├── preproc_config.py          # (model + preproc) version bundle
│       │   └── benchmarks/                # Matcher / dispatch micro-benchmarks
│       └── models/
│           ├── baseline_gait_model.pth    # Trained ResNet-18 weights (CASIA-B, v1)
│           └── *.config.json              # Sidecar config for v2 models (auto-generated)
│
├── templates/               # Jinja2 HTML templates (Tailwind CSS)
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── register_gait.html
│   ├── identify.html
│   ├── identify_gait.html
│   ├── fusion.html
│   ├── attendance.html
│   └── users.html
│
├── static/
│   ├── css/style.css
│   └── js/webcam.js
│
├── database/
│   └── trace.db             # Auto-created SQLite database
│
├── uploads/                 # Temporary directory — videos deleted after inference
├── attendance.csv           # Auto-generated historical check-in log
└── yolov8n-seg.pt           # YOLOv8 segmentation weights
```

---

## Privacy & Security

TRACE is a **privacy-first** system. It stores only mathematical vectors (embeddings) in the database — never raw photos or video frames. All media files are processed in memory or temporary directories and are **immediately deleted after inference**. No biometric data is transmitted to external servers or the cloud.

---

## Authors

**Lovely Professional University — School of AI and Emerging Technologies**

- Ravulapally Anurag Sharma
- Pentakota Hemanth Sai Kumar
- Lokesh Modi
- Karanbir Singh