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
        embedding BLOB
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
SIM_THRESHOLD = 0.6

# -----------------------------
# Helper functions
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

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
        cap = cv2.VideoCapture(0)
        embedding = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                detection = results.detections[0]
                ih, iw, _ = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                face_img = frame[max(0,y1):y1+h, max(0,x1):x1+w]
                emb = get_embedding(face_img)
                if emb is not None:
                    embedding = emb
                    break
        cap.release()
        cv2.destroyAllWindows()

        if embedding is None:
            flash("No face detected. Try again.",'danger')
            return redirect(url_for('register'))

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # check if exists
        c.execute("SELECT name, embedding FROM users")
        rows = c.fetchall()
        for db_name, db_emb_bytes in rows:
            db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
            sim = cosine_similarity(embedding, db_emb)
            if sim > SIM_THRESHOLD:
                flash(f"Face already exists as {db_name}",'danger')
                return redirect(url_for('register'))
            elif name == db_name:
                flash(f"Name {name} already exists.",'danger')
                return redirect(url_for('register'))

        embedding_bytes = embedding.astype(np.float32).tobytes()
        c.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, embedding_bytes))
        conn.commit()
        conn.close()
        flash(f"{name} registered successfully!",'success')
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
