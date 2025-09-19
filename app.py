from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import sqlite3
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
import time

app = Flask(__name__)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

DB_PATH = "entry_exit.db"

# ==============================
# Database
# ==============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    embedding BLOB
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    camera TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

def save_embedding(name, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO persons (name, embedding) VALUES (?, ?)", 
              (name, embedding.tobytes()))
    conn.commit()
    conn.close()

def load_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM persons")
    rows = c.fetchall()
    conn.close()
    names, embeddings = [], []
    for name, emb in rows:
        names.append(name)
        embeddings.append(np.frombuffer(emb, dtype=np.float32))
    return names, np.array(embeddings)

def log_event(name, camera):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch last event for this person
    c.execute("SELECT camera FROM logs WHERE name=? ORDER BY id DESC LIMIT 1", (name,))
    row = c.fetchone()

    if row:
        last_camera = row[0]

        # Person already inside → block repeated ENTRY
        if last_camera == "ENTRY" and camera == "ENTRY":
            print(f"⚠️ Skipping duplicate ENTRY for {name}")
            conn.close()
            return

        # Person already outside → block repeated EXIT
        if last_camera == "EXIT" and camera == "EXIT":
            print(f"⚠️ Skipping duplicate EXIT for {name}")
            conn.close()
            return

    # Save event if valid
    c.execute("INSERT INTO logs (name, camera, timestamp) VALUES (?, ?, ?)", 
              (name, camera, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    print(f"✅ Logged {camera} for {name}")


def fetch_logs(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, camera, timestamp FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# Global Control
# ==============================
frames = {"ENTRY": None, "EXIT": None}
threads = []
stop_flag = False
mode = "dashboard"

# ==============================
# Camera Threads
# ==============================
def camera_thread(cam_id, camera_name):
    global stop_flag
    cap = cv2.VideoCapture(cam_id)
    names, embeddings = load_embeddings()

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = face_app.get(frame)
        for face in faces:
            emb = face.embedding.astype(np.float32)
            best_score, best_name = 0, "Unknown"
            for db_name, db_emb in zip(names, embeddings):
                sim = cosine_similarity(emb, db_emb)
                if sim > best_score:
                    best_score, best_name = sim, db_name
            if best_score > 0.6:  # recognition threshold
                log_event(best_name, camera_name)
                cv2.putText(frame, f"{best_name} {camera_name}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frames[camera_name] = frame

    cap.release()

def start_dashboard():
    global threads, stop_flag, mode
    stop_all_cameras()
    mode = "dashboard"
    stop_flag = False
    threads = [
        threading.Thread(target=camera_thread, args=(0, "ENTRY"), daemon=True),
        threading.Thread(target=camera_thread, args=(2, "EXIT"), daemon=True)
    ]
    for t in threads:
        t.start()

def stop_all_cameras():
    global stop_flag, threads
    stop_flag = True
    time.sleep(0.5)  # let threads close
    threads = []

# ==============================
# Registration Page
# ==============================
@app.route("/register", methods=["GET", "POST"])
def register():
    global mode
    stop_all_cameras()
    mode = "register"

    if request.method == "POST":
        name = request.form["name"]
        cap = cv2.VideoCapture(0)  # dedicated register camera
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Camera error!"

        faces = face_app.get(frame)
        if faces:
            emb = faces[0].embedding.astype(np.float32)

            # Check duplicates before saving
            db_names, db_embeddings = load_embeddings()
            for db_name, db_emb in zip(db_names, db_embeddings):
                sim = cosine_similarity(emb, db_emb)
                if sim > 0.6:  # duplicate threshold
                    return f"❌ Person already exists as {db_name} (similarity={sim:.2f})"

            save_embedding(name, emb)
            return f"✅ {name} registered successfully!"
        else:
            return "❌ No face detected. Try again."
    return render_template("register.html")

# ==============================
# Video Feeds
# ==============================
def generate(camera_name):
    while True:
        if mode != "dashboard":
            break
        frame = frames[camera_name]
        if frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/entry_feed")
def entry_feed():
    return Response(generate("ENTRY"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/exit_feed")
def exit_feed():
    return Response(generate("EXIT"), mimetype="multipart/x-mixed-replace; boundary=frame")

# ==============================
# Logs API
# ==============================
@app.route("/logs")
def logs():
    data = fetch_logs(10)
    return jsonify(data)

# ==============================
# Dashboard
# ==============================
@app.route("/")
def dashboard():
    start_dashboard()
    return render_template("dashboard.html")

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
