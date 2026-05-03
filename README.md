# TRACE — Time & Recognition Automated Check-in Engine

> **Multimodal Biometric Attendance System** — Face Recognition + Gait Recognition + Adaptive Score-Level Fusion

---

## Overview

TRACE is a capstone research project that demonstrates automated, contactless attendance using two biometric modalities fused together:

| Module | Technology | Output |
|--------|-----------|--------|
| Face Recognition | DeepFace + ArcFace (512-D embedding) | `score_face` |
| Gait Recognition | YOLOv8 → GEI → ResNet-18 (512-D embedding) | `score_gait` |
| Fusion Engine | Adaptive score-level fusion | `score_final` |

The fusion adapts weights based on face visibility:
```
If face is clear:    score_final = 0.7 × score_face + 0.3 × score_gait
If face is occluded: score_final = 0.3 × score_face + 0.7 × score_gait
```

---

## Project Structure

```
TRACE_FINAL/
├── app.py                  # Main Flask application (all routes)
├── face_utils.py           # Face embedding + matching utilities
├── gait_utils.py           # Gait GEI extraction + embedding utilities
├── models.py               # SQLAlchemy DB models (User with face+gait)
├── requirements.txt
├── modules/
│   └── gait/
│       ├── src/
│       │   ├── phase1_video_to_gei.py        # YOLO silhouette extraction
│       │   └── phase3_dataset_and_model.py   # ResNet-18 GEI model
│       └── models/
│           └── baseline_gait_model.pth       # Trained model weights
├── templates/
│   ├── base.html           # Dark Tailwind UI shell + nav
│   ├── index.html          # Dashboard with live stats
│   ├── register.html       # Face registration (webcam)
│   ├── identify.html       # Face identification
│   ├── register_gait.html  # Gait registration (video upload)
│   ├── identify_gait.html  # Gait identification
│   ├── fusion.html         # ⚡ Fusion attendance (both modalities)
│   ├── attendance.html     # Attendance log with all scores
│   └── users.html          # User management
├── static/
│   ├── css/style.css
│   └── js/webcam.js
├── database/               # SQLite DB (auto-created)
├── uploads/                # Temp video uploads (auto-created)
└── docs/results/           # Generated results/screenshots
```

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd TRACE_FINAL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Usage

### 1. Register a User (Face)
- Go to **Register Face** → enter name + department
- Capture 5–10 webcam frames
- Face embedding stored in SQLite

### 2. Register Gait
- Go to **Register Gait** → select the registered user
- Upload a 3–8 second walking video (side view preferred)
- GEI extracted → 512-D embedding stored

### 3. Take Attendance
| Mode | Page | How |
|------|------|-----|
| Face only | `/identify` | Webcam snapshot → ArcFace match |
| Gait only | `/identify-gait` | Upload walking video |
| **Fusion** | `/fusion` | Face snapshot + gait video → adaptive fusion |

### 4. View Logs
- `/attendance` — all records with Face/Gait/Final scores + method badge
- Download CSV button for export

---

## Key Results

| Method | Condition | Performance |
|--------|-----------|-------------|
| Face Recognition | Clear, frontal | High accuracy (>80% conf.) |
| Gait Recognition | Side/full-body walk | Stable across distances |
| **TRACE Fusion** | Mixed / occluded | **Best robustness** |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/register` | POST (JSON) | Register face from base64 images |
| `/api/identify` | POST (JSON) | Identify from base64 image |
| `/api/register-gait` | POST (form) | Upload video → register gait |
| `/api/identify-gait` | POST (form) | Upload video → gait identification |
| `/api/identify-fusion` | POST (form) | Face image + gait video → fusion |
| `/api/users` | GET | List all users |
| `/api/users/<id>` | DELETE | Delete user |
| `/api/attendance/download` | GET | Download attendance CSV |

---

## System Requirements

- Python 3.10
- Webcam (for face registration/identification)
- GPU optional but recommended for gait (falls back to CPU)
- 4 GB RAM minimum

---

## Authors

- Ravulapally Anurag Sharma
- Pentakota Hemanth Sai Kumar
- Lokesh Modi
- Karanbir Singh

**Lovely Professional University — School of AI and Emerging Technologies**
