# TRACE — Time & Recognition Automated Check-in Engine

> **Multimodal Biometric Attendance System**
> Face Recognition · Gait Recognition · Adaptive Score-Level Fusion

TRACE is a B.Tech Capstone research project demonstrating automated, contactless attendance using two biometric modalities fused together. Built with Flask, DeepFace, YOLOv8, and PyTorch, it ensures robust identification even when a user's face is partially occluded.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Open-Set Confidence Punisher](#open-set-confidence-punisher)
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

---

## Standard Operating Procedures

The gait pipeline relies on accurate silhouette extraction (Gait Energy Images). Users **must** follow these rules during both registration and identification:

**1. 🎒 The Backpack Rule**
Remove heavy backpacks or oversized coats. YOLOv8 segments based on anatomical shape — external objects create artificial blobs that corrupt the GEI.

**2. 📐 The Camera Angle Rule**
Record from a **90° side profile** (walking left-to-right or right-to-left). Front-facing videos hide stride mechanics and produce near-identical GEIs across users.

**3. 📱 The Orientation Rule**
Both enrollment and inference videos **must be shot in Landscape mode**. Portrait mode distorts the aspect ratio during the 64×64 neural network resize, causing false negatives.

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
│       ├── src/             # GEI generation & dataset/model definitions
│       └── models/
│           └── baseline_gait_model.pth   # Trained ResNet-18 weights (CASIA-B)
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