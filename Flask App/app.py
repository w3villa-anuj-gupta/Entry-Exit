import os
import sqlite3
import cv2
import mediapipe as mp
import torch
import numpy as np
from numpy.linalg import norm
from facenet_pytorch import InceptionResnetV1
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, flash,jsonify
import time
from datetime import datetime

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecret"
DB_NAME = "face.db"

# Camera on and off 

running = True
cap_entry = None
cap_exit = None

# -----------------------------
# Database setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        name TEXT PRIMARY KEY,
        embedding BLOB,
        samples_count INT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        event TEXT,
        timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# FaceNet + Mediapipe setup
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
SIM_THRESHOLD = 0.7

# -----------------------------
# Helper functions
# -----------------------------
def l2_normalize(x, eps=1e-10):
    norm1 = norm(x)
    if norm1 < eps:
        return x
    return x / norm1

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def average_embeddings(emb_list, remove_outliers=True, outlier_thresh=0.5):
    """
    emb_list: list of 1D numpy arrays (float32)
    remove_outliers: remove samples whose cosine to initial mean < outlier_thresh
    Returns: normalized averaged embedding (float32) or None if nothing valid
    """
    if not emb_list:
        return None

    # stack
    E = np.stack(emb_list, axis=0)  # shape (N, d)

    # initial mean
    mean = np.mean(E, axis=0)

    if remove_outliers and len(emb_list) > 2:
        sims = (E @ mean) / (np.linalg.norm(E, axis=1) * np.linalg.norm(mean) + 1e-10)
        # keep embeddings with sim >= outlier_thresh
        keep_idx = np.where(sims >= outlier_thresh)[0]
        if len(keep_idx) == 0:
            # all outliers — fallback to use all
            mean = np.mean(E, axis=0)
        else:
            mean = np.mean(E[keep_idx], axis=0)

    mean = mean.astype(np.float32)
    mean = l2_normalize(mean)
    return mean

# --- capture multiple samples from camera ---
def capture_embeddings_from_camera(cap, face_detection, resnet, samples=12, timeout=12):
    """
    Try to capture up to `samples` good face embeddings within `timeout` seconds.
    Returns list of numpy arrays or empty list.
    Assumes get_embedding(face_img) exists and returns numpy array.
    """
    embeddings = []
    start = time.time()
    while len(embeddings) < samples and (time.time() - start) < timeout:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if results.detections:
            # pick largest face or first detection
            # here: use first detection
            det = results.detections[0]
            ih, iw, _ = frame.shape
            bboxC = det.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            face_img = frame[max(0,y1):y1+h, max(0,x1):x1+w]
            if face_img.size == 0:
                continue
            emb = get_embedding(face_img)  # your function
            if emb is not None:
                embeddings.append(emb.astype(np.float32))
        # small delay to avoid hammering
        cv2.waitKey(50)
    return embeddings


def get_embedding(face_img):
    if face_img.size == 0:
        return None
    face_tensor = torch.tensor(face_img.transpose((2,0,1)), dtype=torch.float32)
    face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0), size=(160,160))
    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        emb = resnet(face_tensor).squeeze().numpy()
    return emb

def identify_face(face_img):
    emb = get_embedding(face_img)
    if emb is None:
        return None
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM users")
    rows = c.fetchall()
    conn.close()
    for db_name, db_emb_bytes in rows:
        db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
        sim = cosine_similarity(emb, db_emb)
        if sim > SIM_THRESHOLD:
            return db_name
    return None

def log_event(name, event):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO logs (name, event, timestamp) VALUES (?, ?, ?)", (name, event, timestamp))
    conn.commit()
    conn.close()
    return timestamp

def fetch_logs(limit=10):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, event, timestamp FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------------
# Global state for tracking
# -----------------------------
person_status = {}       # {name: "Inside"/"Outside"}
last_exit_time = {}      # {name: datetime}
COOLDOWN = 10            # seconds

# -----------------------------
# Camera generators
# -----------------------------
def gen_frames(cam_id, cam_type):
    global running, cap_entry, cap_exit
    if cam_type == "entry":
        cap_entry = cv2.VideoCapture(cam_id)
        cap = cap_entry
    else:
        cap_exit = cv2.VideoCapture(cam_id)
        cap = cap_exit
    while running:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                ih, iw, _ = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                face_img = frame[max(0,y1):y1+h, max(0,x1):x1+w]
                name = identify_face(face_img)
                if name:
                    now = datetime.now()
                    status = person_status.get(name, "Outside")
                    if cam_type == "entry":
                        last_exit = last_exit_time.get(name)
                        if status == "Outside" and (last_exit is None or (now - last_exit).total_seconds() >= COOLDOWN):
                            ts = log_event(name, "Entry")
                            person_status[name] = "Inside"
                            cv2.putText(frame, f"{name} ENTERED {ts}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    else:  # exit camera
                        if status == "Inside":
                            ts = log_event(name, "Exit")
                            person_status[name] = "Outside"
                            last_exit_time[name] = now
                            cv2.putText(frame, f"{name} EXITED {ts}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/entry_camera")
def entry_camera():
    return Response(gen_frames(0, "entry"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/exit_camera")
def exit_camera():
    return Response(gen_frames(2, "exit"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/entry_exit")
def entry_exit():
    global running
    running = True   # restart cameras when page loads
    return render_template("entry_exit.html")

@app.route("/stop_cameras")
def stop_cameras():
    global running, cap_entry, cap_exit
    running = False
    if cap_entry is not None:
        cap_entry.release()
        cap_entry = None
    if cap_exit is not None:
        cap_exit.release()
        cap_exit = None
    return redirect(url_for("home"))


@app.route("/logs")
def logs():
    data = fetch_logs(10)
    return jsonify(data)

# -----------------------------
# Register Face
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        if not name:
            flash("Name is required.", "danger")
            return redirect(url_for("register"))

        cap = cv2.VideoCapture(0)
        try:
            # capture multiple embeddings (e.g., 6 samples within 12s)
            raw_embs = capture_embeddings_from_camera(cap, face_detection, resnet, samples=12, timeout=12)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not raw_embs:
            flash("Could not detect face. Try again.", "danger")
            return redirect(url_for("register"))

        # compute averaged normalized embedding with outlier removal
        avg_emb = average_embeddings(raw_embs, remove_outliers=True, outlier_thresh=0.5)
        if avg_emb is None:
            flash("Failed to compute averaged embedding.", "danger")
            return redirect(url_for("register"))

        # check duplicates (compare avg_emb against DB)
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, embedding FROM users")
        rows = c.fetchall()
        for db_name, db_emb_bytes in rows:
            db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
            sim = cosine_similarity(avg_emb, db_emb)
            if sim > SIM_THRESHOLD:
                flash(f"Face already exists as {db_name} (sim={sim:.2f})", 'danger')
                conn.close()
                return redirect(url_for("register"))
            if name == db_name:
                flash(f"Name {name} already exists.", 'danger')
                conn.close()
                return redirect(url_for("register"))

        # store averaged embedding
        emb_bytes = avg_emb.astype(np.float32).tobytes()
        c.execute("INSERT INTO users (name, embedding, samples_count) VALUES (?, ?, ?)", (name, emb_bytes, len(raw_embs)))
        conn.commit()
        conn.close()
        flash(f"{name} registered successfully!", 'success')

    return render_template("register.html")
# -----------------------------
# Delete Face
# -----------------------------
@app.route("/delete", methods=["GET", "POST"])
def delete():
    if request.method == "POST":
        name = request.form["name"].strip()
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, embedding FROM users")
        rows = c.fetchall()
        for db_name,embeddings in rows:
            if db_name == name:
                c.execute('''
                DELETE FROM users WHERE name = '{}';
                '''.format(name))
                conn.commit()
                flash(f"{name} deleted successfully",'success')
                break
        else:
            flash(f"{name} Not Exist ",'danger')
        conn.close()
        
    return render_template("delete.html")

# ----------------------------
# Display Users 
# ----------------------------
@app.route("/display_users",methods=["GET"])
def display_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM users")
    rows = c.fetchall()
    conn.close()
    return render_template('display_users.html',users=rows)

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)