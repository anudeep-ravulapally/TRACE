# app.py  –  TRACE: Time & Recognition Automated Check-in Engine
# Unified Flask backend: Face + Gait + Adaptive Fusion
import os
import csv
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from models import db, User
from face_utils import base64_to_image, get_embedding, get_averaged_embedding, find_best_match
from gait_utils import get_gait_embedding_from_video, find_best_gait_match

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
CSV_PATH     = os.path.join(BASE_DIR, "attendance.csv")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(DATABASE_DIR, "trace.db")
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB upload limit for gait videos
db.init_app(app)

with app.app_context():
    db.create_all()


# ── CSV Helpers ────────────────────────────────────────────────────────
# Updated headers to support multimodal scoring
CSV_HEADERS = ["Name", "Department", "Date", "Time",
               "Face Score", "Gait Score", "Final Score", "Method"]

def log_attendance(name, dept, face_score, gait_score, final_score, method):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADERS)
        now = datetime.now()
        writer.writerow([
            name, dept,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            f"{face_score:.2f}"  if face_score  is not None else "—",
            f"{gait_score:.2f}"  if gait_score  is not None else "—",
            f"{final_score:.2f}" if final_score is not None else "—",
            method
        ])

def read_attendance():
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reversed(list(reader)))


# ── Fusion Engine ──────────────────────────────────────────────────────
def adaptive_fusion(face_score, gait_score, face_detected=True):
    """
    Adaptive score-level fusion.
      - If face is clearly visible (high confidence) → trust face more
      - If face occluded / low confidence          → lean on gait
    Returns final_score in [0, 1].
    """
    if face_score is None and gait_score is None:
        return 0.0
    if face_score is None:
        return gait_score
    if gait_score is None:
        return face_score

    # Adaptive weight: face quality proxy = face_score itself
    if face_detected and face_score >= 0.5:
        alpha, beta = 0.7, 0.3      # Clear face → trust face
    else:
        alpha, beta = 0.3, 0.7      # Occluded   → trust gait

    return alpha * face_score + beta * gait_score


# ── Page Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    records = read_attendance()
    today   = datetime.now().strftime("%Y-%m-%d")
    today_checkins = sum(1 for r in records if r.get("Date") == today)
    total_checkins = len(records)

    scores = []
    for r in records:
        try:
            scores.append(float(r["Final Score"]))
        except (KeyError, ValueError):
            try:
                scores.append(float(r.get("Face Score", 0)))
            except Exception:
                pass

    avg_confidence = round(sum(scores) / len(scores) * 100, 1) if scores else 0

    stats = {
        "total_users":    User.query.count(),
        "today_checkins": today_checkins,
        "total_checkins": total_checkins,
        "avg_confidence": avg_confidence,
        "recent_records": records[:5],
    }
    return render_template("index.html", active_page="dashboard", stats=stats)

@app.route("/register")
def register_page():
    return render_template("register.html", active_page="register")

@app.route("/identify")
def identify_page():
    return render_template("identify.html", active_page="identify")

@app.route("/register-gait")
def register_gait_page():
    users = User.query.order_by(User.full_name).all()
    return render_template("register_gait.html", active_page="register_gait", users=users)

@app.route("/identify-gait")
def identify_gait_page():
    return render_template("identify_gait.html", active_page="identify_gait")

@app.route("/fusion")
def fusion_page():
    return render_template("fusion.html", active_page="fusion")

@app.route("/attendance")
def attendance_page():
    records = read_attendance()
    return render_template("attendance.html", records=records, active_page="attendance")

@app.route("/users")
def users_page():
    return render_template("users.html")


# ── API: Face Register ─────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data      = request.get_json()
    full_name = data.get("full_name", "").strip()
    department = data.get("department", "General").strip()
    images    = data.get("images")

    if not full_name:
        return jsonify({"error": "Full name is required"}), 400
    if not images or not isinstance(images, list) or len(images) == 0:
        return jsonify({"error": "At least one image is required"}), 400

    try:
        embedding = get_averaged_embedding(images)
        # Update existing user or create new
        user = User.query.filter_by(full_name=full_name).first()
        if user:
            user.set_embedding(embedding)
            user.department = department
        else:
            user = User(full_name=full_name, department=department)
            user.set_embedding(embedding)
            db.session.add(user)
        db.session.commit()
        return jsonify({
            "success": True,
            "message": f"{full_name} face registered! ({len(images)} frames averaged)"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Face Identify ─────────────────────────────────────────────────
@app.route("/api/identify", methods=["POST"])
def identify():
    data      = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "Image is required"}), 400

    try:
        image        = base64_to_image(image_b64)
        embedding    = get_embedding(image)
        all_users    = User.query.filter(User.embedding.isnot(None)).all()
        match, score = find_best_match(embedding, all_users)

        if match:
            log_attendance(match.full_name, match.department or "—",
                           score, None, score, "Face")
            return jsonify({
                "identified": True,
                "name":       match.full_name,
                "confidence": round(score * 100, 2),
                "method":     "Face"
            })
        else:
            return jsonify({"identified": False, "message": "Face not recognized",
                            "confidence": round(score * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Gait Register ─────────────────────────────────────────────────
@app.route("/api/register-gait", methods=["POST"])
def register_gait():
    """
    Accepts one or more video file uploads.
    Extracts GEI → embedding for each → stores in user record.

    Form fields:
        user_id : int        — required
        video   : file       — single-clip enrollment (form field "video")
        videos  : file[]     — multi-clip enrollment (form field "videos",
                                repeat the field name once per clip)
        append  : "true"|... — if truthy, ADD these clips to whatever the
                                user already has enrolled. Default replaces.
    """
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Collect clip files: support both "video" (single) and "videos" (multi).
    files = request.files.getlist("videos")
    single = request.files.get("video")
    if single:
        files = list(files) + [single]
    if not files:
        return jsonify({"error": "At least one video file is required"}), 400

    append = str(request.form.get("append", "")).lower() in ("1", "true", "yes")

    # Extract embedding for each clip; keep going on per-clip failures so a
    # single bad clip doesn't doom a multi-clip enrollment.
    new_embeddings = []
    errors = []
    tmp_paths = []
    for i, video in enumerate(files):
        suffix = os.path.splitext(video.filename)[1] or ".mp4"
        tmp_path = os.path.join(UPLOAD_DIR, f"gait_reg_{user_id}_{i}{suffix}")
        video.save(tmp_path)
        tmp_paths.append(tmp_path)
        try:
            new_embeddings.append(get_gait_embedding_from_video(tmp_path))
        except ValueError as e:
            errors.append(f"clip {i+1}: {e}")
        except Exception as e:
            errors.append(f"clip {i+1}: {e}")

    for p in tmp_paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    if not new_embeddings:
        return jsonify({"error": "; ".join(errors) or "No usable clips."}), 400

    try:
        if append:
            for emb in new_embeddings:
                user.add_gait_embedding(emb)
        else:
            user.set_gait_embeddings(new_embeddings)
        db.session.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    msg = f"Gait registered for {user.full_name}! ({len(new_embeddings)} clip(s))"
    if errors:
        msg += " Some clips were skipped: " + "; ".join(errors)
    return jsonify({"success": True, "message": msg, "clips": len(new_embeddings)})

# ── API: Gait Identify ─────────────────────────────────────────────────
@app.route("/api/identify-gait", methods=["POST"])
def identify_gait():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "Video file is required"}), 400

    suffix   = os.path.splitext(video.filename)[1] or ".mp4"
    tmp_path = os.path.join(UPLOAD_DIR, f"gait_id_{datetime.now().strftime('%H%M%S')}{suffix}")
    video.save(tmp_path)

    try:
        from gait_utils import vanity_score
        embedding = get_gait_embedding_from_video(tmp_path)
        all_users = User.query.filter(User.gait_embedding.isnot(None)).all()
        match, scaled, raw = find_best_gait_match(embedding, all_users)
        os.remove(tmp_path)

        display_pct = round(vanity_score(scaled) * 100, 2)

        if match:
            log_attendance(match.full_name, match.department or "—",
                           None, scaled, scaled, "Gait")
            return jsonify({
                "identified": True,
                "name":       match.full_name,
                "confidence": display_pct,
                "raw_score":  round(raw * 100, 4),
                "method":     "Gait"
            })
        else:
            return jsonify({
                "identified": False,
                "message":    "Gait not recognized",
                "confidence": round(scaled * 100, 2),
                "raw_score":  round(raw * 100, 4)
            })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Fusion Identify ───────────────────────────────────────────────
@app.route("/api/identify-fusion", methods=["POST"])
def identify_fusion():
    """
    Multimodal fusion attendance.
    """
    image_b64 = request.form.get("image")
    video     = request.files.get("video")

    if not image_b64 and not video:
        return jsonify({"error": "Provide at least a face image or gait video"}), 400

    face_match = face_score = None
    gait_match = gait_score = gait_raw = None
    face_detected = False

    # ── Face arm ──────────────────────────────────────────────────────
    if image_b64:
        try:
            img  = base64_to_image(image_b64)
            emb  = get_embedding(img)
            all_face_users = User.query.filter(User.embedding.isnot(None)).all()
            face_match, face_score = find_best_match(emb, all_face_users)
            face_detected = True
        except Exception:
            face_detected = False
            face_score    = None

    # ── Gait arm ──────────────────────────────────────────────────────
    if video:
        suffix   = os.path.splitext(video.filename)[1] or ".mp4"
        tmp_path = os.path.join(UPLOAD_DIR,
                                f"fusion_{datetime.now().strftime('%H%M%S%f')}{suffix}")
        video.save(tmp_path)
        try:
            g_emb = get_gait_embedding_from_video(tmp_path)
            all_gait_users = User.query.filter(User.gait_embedding.isnot(None)).all()
            gait_match, gait_score, gait_raw = find_best_gait_match(g_emb, all_gait_users)
        except Exception:
            gait_score = 0.0
            gait_raw   = 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Fusion decision ───────────────────────────────────────────────
    final_score = adaptive_fusion(face_score, gait_score, face_detected)
    FUSION_THRESHOLD = 0.50

    candidate = None
    if face_match and gait_match:
        candidate = face_match if (face_score or 0) >= (gait_score or 0) else gait_match
    elif face_match:
        candidate = face_match
    elif gait_match:
        candidate = gait_match

    method_parts = []
    if image_b64:
        method_parts.append("Face")
    if video:
        method_parts.append("Gait")
    method = "+".join(method_parts) + " Fusion"

    if candidate and final_score >= FUSION_THRESHOLD:
        from gait_utils import vanity_score
        gait_display = round(vanity_score(gait_score or 0.0) * 100, 2) if gait_score else 0.0
        log_attendance(
            candidate.full_name, candidate.department or "—",
            face_score, gait_score, final_score, method
        )
        return jsonify({
            "identified":   True,
            "name":         candidate.full_name,
            "face_score":   round((face_score or 0) * 100, 2),
            "gait_score":   gait_display,
            "final_score":  round(final_score * 100, 2),
            "alpha":        0.7 if (face_detected and (face_score or 0) >= 0.5) else 0.3,
            "beta":         0.3 if (face_detected and (face_score or 0) >= 0.5) else 0.7,
            "method":       method
        })
    else:
        return jsonify({
            "identified":  False,
            "message":     "Identity could not be confirmed",
            "face_score":  round((face_score or 0) * 100, 2),
            "gait_score":  round((gait_score or 0) * 100, 2),
            "final_score": round(final_score * 100, 2),
            "method":      method
        })

# ── API: Users CRUD ────────────────────────────────────────────────────
@app.route("/api/users", methods=["GET"])
def get_users():
    query = request.args.get("q", "").strip()
    if query:
        users = User.query.filter(User.full_name.ilike(f"%{query}%")).all()
    else:
        users = User.query.order_by(User.full_name).all()
    return jsonify([{
        "id":         u.id,
        "name":       u.full_name,
        "department": u.department or "—",
        "has_face":   u.has_face(),
        "has_gait":   u.has_gait()
    } for u in users])

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    db.session.delete(user)
    db.session.commit()
    return jsonify({"success": True, "message": f"{user.full_name} deleted."})


# ── API: Download CSV ──────────────────────────────────────────────────
@app.route("/api/attendance/download")
def download_csv():
    if not os.path.isfile(CSV_PATH):
        return jsonify({"error": "No attendance records yet."}), 404
    return send_file(CSV_PATH, as_attachment=True, download_name="attendance.csv")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)