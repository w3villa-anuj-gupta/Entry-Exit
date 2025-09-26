import os
import sqlite3
import cv2
import mediapipe as mp
import torch
import numpy as np
from numpy.linalg import norm
from facenet_pytorch import InceptionResnetV1
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import base64
from io import BytesIO
from PIL import Image
import faiss
import time

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecret"
DB_NAME = "face.db"
EMBEDDING_SIZE = 512
SIM_THRESHOLD = 0.6
COOLDOWN = 10  # seconds

# Global state
index = None
ids, names = [], []
running_entry_exit = True
running_register = True
cap_entry = None
cap_exit = None
cap_register = None
person_status = {}
last_exit_time = {}

# -----------------------------
# Database setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
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

def load_faiss_index():
    global index, ids, names
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM users")
    rows = c.fetchall()
    conn.close()

    # ID-mapped index
    index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_SIZE))
    ids.clear()
    names.clear()

    for db_id, name, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != EMBEDDING_SIZE:
            print(f"⚠️ Skipping {name}, wrong dim {emb.shape[1]}")
            continue
        index.add_with_ids(emb, np.array([db_id], dtype=np.int64))
        ids.append(db_id)
        names.append(name)

init_db()
load_faiss_index()

# -----------------------------
# FaceNet + Mediapipe setup
# -----------------------------
mp_face = mp.solutions.face_detection
# Use separate FaceDetection objects for each camera
face_detection_entry = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_detection_exit = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

# -----------------------------
# Helper functions
# -----------------------------
def get_embedding(face_img):
    if face_img.size == 0:
        return None
    face_tensor = torch.tensor(face_img.transpose((2,0,1)), dtype=torch.float32)
    face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0), size=(160,160))
    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        emb = resnet(face_tensor).squeeze().numpy()
    return emb

def identify_face(embedding):
    global index, names, ids
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(embedding, k=5)
    matched_names = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        if distances[0][rank] < SIM_THRESHOLD:
            # idx is the ID from SQLite
            matched_names.append(names[ids.index(idx)])
    if not matched_names:
        return None
    print("matched_names : ",matched_names)
    return max(set(matched_names), key=matched_names.count)


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
# Camera generators
# -----------------------------
def gen_frames(cam_id, cam_type):
    global running_entry_exit, cap_entry, cap_exit
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if cam_type == "entry":
        cap_entry = cap
        face_detector = face_detection_entry
    else:
        cap_exit = cap
        face_detector = face_detection_exit

    while running_entry_exit:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # Ensure contiguous frame
        frame = np.ascontiguousarray(frame)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb_frame is None or rgb_frame.size == 0:
            continue

        # Face detection
        try:
            results = face_detector.process(rgb_frame)
        except Exception as e:
            print(f"{cam_type} : ⚠️ Mediapipe processing error:", e)
            continue

        if results.detections:
            for detection in results.detections:
                ih, iw, _ = frame.shape
                bbox = detection.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin*iw), int(bbox.ymin*ih)
                w, h = int(bbox.width*iw), int(bbox.height*ih)
                face_img = frame[max(0,y1):y1+h, max(0,x1):x1+w]
                if face_img is None or face_img.size == 0:
                    continue
                emb = get_embedding(face_img)
                name = identify_face(emb) if emb is not None else None

                if name:
                    now = datetime.now()
                    status = person_status.get(name, "Outside")

                    if cam_type == "entry":
                        last_exit = last_exit_time.get(name)
                        if status=="Outside" and (last_exit is None or (now-last_exit).total_seconds()>=COOLDOWN):
                            ts = log_event(name, "Entry")
                            person_status[name] = "Inside"
                            cv2.putText(frame, f"{name} ENTERED {ts}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
                    else:
                        if status=="Inside":
                            ts = log_event(name, "Exit")
                            person_status[name] = "Outside"
                            last_exit_time[name] = now
                            cv2.putText(frame, f"{name} EXITED {ts}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def gen_register_camera():
    global running_register, cap_register
    cap_register = cv2.VideoCapture(0)
    cap_register.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_register.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap = cap_register

    while running_register:
        if not cap.isOpened():
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue
        success, frame = cap.read()
        if not success or frame is None or frame.size==0:
            time.sleep(0.01)
            continue

        frame = np.ascontiguousarray(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

    cap.release()
    cap_register=None

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

@app.route("/register_camera_feed")
def register_camera_feed():
    global running_register
    running_register = True
    return Response(gen_register_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/entry_exit")
def entry_exit():
    global running_entry_exit
    running_entry_exit=True
    return render_template("entry_exit.html")

@app.route("/stop_cameras")
def stop_cameras():
    global running_entry_exit, running_register, cap_entry, cap_exit, cap_register
    running_entry_exit = False
    running_register = False
    for c in [cap_entry, cap_exit, cap_register]:
        if c is not None:
            c.release()
    cap_entry = cap_exit = cap_register = None
    return redirect(url_for("home"))

@app.route("/logs")
def logs():
    data = fetch_logs(10)
    return jsonify(data)

# -----------------------------
# Register Face
# -----------------------------
@app.route("/capture_embedding", methods=["POST"])
def capture_embedding():
    data = request.json["image"]
    img_str = data.split(",")[1]
    img_bytes = base64.b64decode(img_str)
    img = np.array(Image.open(BytesIO(img_bytes)))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    try:
        results = face_detection_entry.process(rgb_img)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    if results.detections:
        det = results.detections[0]
        ih, iw, _ = rgb_img.shape
        bbox = det.location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin*iw), int(bbox.ymin*ih)
        w, h = int(bbox.width*iw), int(bbox.height*ih)
        face_img = rgb_img[max(0,y1):y1+h, max(0,x1):x1+w]
        emb = get_embedding(face_img)
        if emb is not None:
            return jsonify({"success": True, "embedding": emb.tolist()})
    return jsonify({"success": False})

@app.route("/register_multiple", methods=["POST"])
def register_multiple():
    global index, names, ids
    data = request.json
    name = data["name"].strip()
    embeddings_list = data["embeddings"]

    if len(embeddings_list)!=5:
        return jsonify({"success": False, "message":"Need 5 embeddings"})

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM users WHERE name=?", (name,))
    if c.fetchone():
        return jsonify({"success":False,"message":f"{name} already exists"})

    new_ids = []
    for emb in embeddings_list:
        emb_np = np.array(emb, dtype=np.float32).reshape(1,-1)
        emb_bytes = emb_np.tobytes()
        c.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb_bytes))
        new_id = c.lastrowid
        # Add embedding to Faiss with its ID
        index.add_with_ids(emb_np, np.array([new_id], dtype=np.int64))
        ids.append(new_id)
        names.append(name)
        new_ids.append(new_id)

    conn.commit()
    conn.close()
    return jsonify({"success": True, "message":f"{name} registered successfully with IDs: {new_ids}"})


@app.route("/register")
def register():
    return render_template("register.html")

# -----------------------------
# Delete Face
# -----------------------------
@app.route("/delete", methods=["GET", "POST"])
def delete():
    global index, ids, names
    if request.method=="POST":
        name = request.form["name"].strip()
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # Get all IDs for this user
        c.execute("SELECT id FROM users WHERE name=?", (name,))
        id_rows = c.fetchall()
        if id_rows:
            ids_to_remove = [row[0] for row in id_rows]
            # Remove embeddings from Faiss
            index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            # Update parallel lists
            ids = [i for i in ids if i not in ids_to_remove]
            names = [n for i,n in zip(ids, names) if i not in ids_to_remove]
            # Delete from SQLite
            c.execute("DELETE FROM users WHERE name=?", (name,))
            c.execute("DELETE FROM logs WHERE name=?", (name,))
            flash(f"{name} deleted successfully",'success')
        else:
            flash(f"{name} does not exist",'danger')
        conn.commit()
        conn.close()
    return render_template("delete.html")


@app.route("/display_users")
def display_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM users")
    rows = c.fetchall()
    conn.close()
    users = [row[0] for row in rows]
    return render_template("display_users.html", users=users)

# -----------------------------
# Run Flask
# -----------------------------
if __name__=="__main__":
    app.run(debug=True)
