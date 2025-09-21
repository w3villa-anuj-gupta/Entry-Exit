import cv2
import threading
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import sqlite3
from datetime import datetime
from numpy.linalg import norm

# -----------------------------
# Global variables
# -----------------------------
frame1, frame2 = None, None
running = True

# Track person status: "Inside" or "Outside"
person_status = {}  # {name: "Inside"/"Outside"}

# -----------------------------
# Cosine similarity function
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# Database setup
# -----------------------------
conn = sqlite3.connect("faces.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    embedding BLOB
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS logs (
    name TEXT,
    event TEXT,
    timestamp TEXT
)
''')
conn.commit()

# -----------------------------
# FaceNet & MediaPipe setup
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
threshold = 0.6  # similarity threshold

# -----------------------------
# Thread function to grab frames
# -----------------------------
def grab_frames(cv2_obj, cam_id):
    global frame1, frame2, running
    while running:
        ret, frame = cv2_obj.read()
        if not ret:
            continue
        if cam_id == 1:
            frame1 = frame
        else:
            frame2 = frame

# -----------------------------
# Identify person by embedding
# -----------------------------
def identify_face(face_img):
    if face_img.size == 0:
        return None
    face_tensor = torch.tensor(face_img.transpose((2,0,1)), dtype=torch.float32)
    face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0), size=(160,160))
    face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        emb = resnet(face_tensor).squeeze().numpy()
    # Compare with DB
    c.execute("SELECT name, embedding FROM users")
    rows = c.fetchall()
    for db_name, db_emb_bytes in rows:
        db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
        sim = cosine_similarity(emb, db_emb)
        if sim > threshold:
            return db_name
    return None

# -----------------------------
# Main display & detection loop
# -----------------------------
def display_frames():
    global running, frame1, frame2, person_status
    while running:
        # ENTRY camera
        if frame1 is not None:
            rgb_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame1.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    face_img = frame1[max(0,y1):y1+h, max(0,x1):x1+w]
                    name = identify_face(face_img)
                    if name:
                        # Only log entry if person is outside
                        status = person_status.get(name, "Outside")
                        if status == "Outside":
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            c.execute("INSERT INTO logs (name,event,timestamp) VALUES (?,?,?)", (name,"Entry",timestamp))
                            conn.commit()
                            person_status[name] = "Inside"
                            print(f"{name} entered at {timestamp}")
            cv2.imshow("Entry Camera", frame1)

        # EXIT camera
        if frame2 is not None:
            rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame2.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    face_img = frame2[max(0,y1):y1+h, max(0,x1):x1+w]
                    name = identify_face(face_img)
                    if name:
                        # Only log exit if person is inside
                        status = person_status.get(name, "Outside")
                        if status == "Inside":
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            c.execute("INSERT INTO logs (name,event,timestamp) VALUES (?,?,?)", (name,"Exit",timestamp))
                            conn.commit()
                            person_status[name] = "Outside"
                            print(f"{name} exited at {timestamp}")
            cv2.imshow("Exit Camera", frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    thread1 = threading.Thread(target=grab_frames, args=(cap1,1))
    thread2 = threading.Thread(target=grab_frames, args=(cap2,2))

    thread1.start()
    thread2.start()

    display_frames()

    thread1.join()
    thread2.join()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    conn.close()
