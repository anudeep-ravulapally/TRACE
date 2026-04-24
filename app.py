import os
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from models import db, User
from face_utils import base64_to_image, get_embedding, get_averaged_embedding, find_best_match

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
CSV_PATH     = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATABASE_DIR, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(DATABASE_DIR, "trace.db")
db.init_app(app)

with app.app_context():
    db.create_all()

# ── CSV Helper ────────────────────────────────────────────────────────
def log_attendance(name, confidence):
    """Append one row to attendance.csv, creating the file if needed."""
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Confidence (%)"])  # header
        now = datetime.now()
        writer.writerow([
            name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            f"{confidence:.2f}"
        ])

def read_attendance():
    """Read all rows from attendance.csv and return as a list of dicts."""
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reversed(list(reader)))   # newest first

# ── Pages ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    records = read_attendance()
    today   = datetime.now().strftime("%Y-%m-%d")

    today_checkins = sum(1 for r in records if r.get("Date") == today)
    total_checkins = len(records)

    confidences = []
    for r in records:
        try:
            confidences.append(float(r["Confidence (%)"]))
        except (KeyError, ValueError):
            pass
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0

    stats = {
        "total_users":    User.query.count(),
        "today_checkins": today_checkins,
        "total_checkins": total_checkins,
        "avg_confidence": avg_confidence,
        "recent_records": records[:5],   # 5 most recent for dashboard preview
    }
    return render_template("index.html", active_page="dashboard", stats=stats)

@app.route("/register")
def register_page():
    return render_template("register.html", active_page="register")

@app.route("/identify")
def identify_page():
    return render_template("identify.html", active_page="identify")

@app.route("/attendance")
def attendance_page():
    records = read_attendance()
    return render_template("attendance.html", records=records, active_page="attendance")

# ── API: Download raw CSV ─────────────────────────────────────────────
@app.route("/api/attendance/download")
def download_csv():
    if not os.path.isfile(CSV_PATH):
        return jsonify({"error": "No attendance records yet."}), 404
    return send_file(CSV_PATH, as_attachment=True, download_name="attendance.csv")

# ── API: Register (multi-frame averaged embedding) ────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data      = request.get_json()
    full_name = data.get("full_name", "").strip()
    images    = data.get("images")   # list of base64 strings

    if not full_name:
        return jsonify({"error": "Full name is required"}), 400
    if not images or not isinstance(images, list) or len(images) == 0:
        return jsonify({"error": "At least one image is required"}), 400

    try:
        # Compute the mean embedding across all captured frames
        embedding = get_averaged_embedding(images)
        user      = User(full_name=full_name)
        user.set_embedding(embedding)
        db.session.add(user)
        db.session.commit()
        return jsonify({
            "success": True,
            "message": f"{full_name} registered successfully! "
                       f"(averaged from {len(images)} frames)"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Identify ─────────────────────────────────────────────────────
@app.route("/api/identify", methods=["POST"])
def identify():
    data      = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "Image is required"}), 400

    try:
        image        = base64_to_image(image_b64)
        embedding    = get_embedding(image)
        all_users    = User.query.all()
        match, score = find_best_match(embedding, all_users)

        if match:
            confidence = round(score * 100, 2)
            log_attendance(match.full_name, confidence)   # ← saves to CSV
            return jsonify({
                "identified": True,
                "name":       match.full_name,
                "confidence": confidence
            })
        else:
            return jsonify({"identified": False, "message": "Face not recognized"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Get all users (with optional search) ─────────────────────────
@app.route("/api/users", methods=["GET"])
def get_users():
    query = request.args.get("q", "").strip()
    if query:
        users = User.query.filter(User.full_name.ilike(f"%{query}%")).all()
    else:
        users = User.query.all()
    return jsonify([{"id": u.id, "name": u.full_name} for u in users])

# ── API: Delete a single user ─────────────────────────────────────────
@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({"success": True, "message": f"{user.full_name} deleted."})

# ── Page: User Management ─────────────────────────────────────────────
@app.route("/users")
def users_page():
    return render_template("users.html")

if __name__ == "__main__":
    app.run(debug=True)