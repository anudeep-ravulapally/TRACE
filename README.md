# 🔍 TRACE — Time & Recognition Automated Check-in Engine

A multi-modal, privacy-first attendance system that uses **face recognition** (and gait, coming soon) to automatically log check-ins. Built with Flask + DeepFace (ArcFace model).

---

## ✨ What It Does

- 📸 **Register** a user's face via webcam (5-expression sequence for accuracy)
- ⚡ **Identify** a user in real-time and log their attendance automatically
- 📋 **View** the full attendance log with confidence scores
- 👥 **Manage** registered users (search + delete)
- ⬇️ **Export** attendance as a CSV file anytime

---

## 🖥️ Prerequisites

Make sure you have these installed before starting:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.9 – 3.11 | [python.org](https://www.python.org/downloads/) |
| pip | (comes with Python) | — |
| Git | any | [git-scm.com](https://git-scm.com/) |
| A webcam | — | (built-in or USB) |

> ⚠️ **Python 3.12+ is not supported** — TensorFlow (used by DeepFace) requires Python 3.9–3.11.

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/trace.git
cd trace
```

> If you received the project as a ZIP, just extract it and open a terminal inside the folder.

---

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv trace_venv
trace_venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv trace_venv
source trace_venv/bin/activate
```

You should see `(trace_venv)` appear at the start of your terminal prompt. ✅

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This will take a few minutes — it installs TensorFlow, DeepFace, OpenCV, and Flask.  
> Make sure you have a stable internet connection.

---

### 4. Run the app

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

### 5. Open in your browser

Visit 👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

> The first launch may be slow (30–60 seconds) as DeepFace downloads the ArcFace model weights automatically.

---

## 🗺️ Pages & How to Use Them

| Page | URL | What to do |
|------|-----|-----------|
| **Dashboard** | `/` | Overview of stats and recent check-ins |
| **Register Face** | `/register` | Enter your name → click Start Camera → follow the 5-expression guide |
| **Identify** | `/identify` | Click Start Camera → click Identify Me |
| **Attendance Log** | `/attendance` | View all check-ins, search by name/date |
| **Manage Users** | `/users` | Search or delete registered users |

### Quick tip for registration:
The system captures 5 expressions (Neutral, Smile, Angry, Squinting, Goofy) — just follow the on-screen prompts and hold each expression for 3 seconds. Better lighting = better accuracy.

---

## 📁 Project Structure

```
trace/
├── app.py              # Flask routes & API endpoints
├── face_utils.py       # ArcFace embedding logic (DeepFace)
├── models.py           # SQLAlchemy User model
├── requirements.txt    # All Python dependencies
├── attendance.csv      # Auto-generated attendance log
├── database/
│   └── trace.db        # SQLite database (auto-created)
├── static/
│   ├── css/style.css
│   └── js/webcam.js
└── templates/
    ├── base.html
    ├── index.html
    ├── register.html
    ├── identify.html
    ├── attendance.html
    └── users.html
```

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError` for deepface / tensorflow**  
→ Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**Camera not working in browser**  
→ Try Chrome or Edge. Firefox sometimes blocks webcam on localhost. Make sure no other app (Zoom, Teams) is using the camera.

**First identification is very slow**  
→ Normal! DeepFace is loading the ArcFace model into memory. Subsequent ones are fast.

**Face not recognized (confidence too low)**  
→ Re-register in better lighting. The threshold is set to 68% cosine similarity — you can tune `THRESHOLD` in `face_utils.py`.

**`pip install` fails on TensorFlow (Windows)**  
→ Make sure you're using Python 3.9–3.11 (not 3.12+). Run `python --version` to check.

---

## 🔒 Privacy Note

TRACE stores **only mathematical vectors** (embeddings), never your actual photos or video. All data stays on your local machine — nothing is sent to the cloud.

---

## 📦 Key Dependencies

- **[Flask](https://flask.palletsprojects.com/)** — Web framework
- **[DeepFace](https://github.com/serengil/deepface)** — Face recognition (ArcFace + RetinaFace)
- **[TensorFlow](https://www.tensorflow.org/)** — Deep learning backend for DeepFace
- **[OpenCV](https://opencv.org/)** — Image processing
- **[Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/)** — Database ORM

---

*TRACE — B.Tech Capstone Project · Phase 2: Face + Gait Recognition*