import sys
import sqlite3
import cv2
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from numpy.linalg import norm

# -----------------------------
# Helper: cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# Check command-line argument
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python register_face.py <name>")
    sys.exit(1)

name = sys.argv[1]
name = name.strip()

# -----------------------------
# Setup SQLite DB
# -----------------------------
conn = sqlite3.connect("faces.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    embedding BLOB
)
''')
conn.commit()

# -----------------------------
# Initialize Face Detection & FaceNet
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# -----------------------------
# Capture face from camera
# -----------------------------
cap = cv2.VideoCapture(2)
print("Looking for a face. Press 'q' to quit.")

embedding = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x1 = int(bboxC.xmin * iw)
        y1 = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + w, y1 + h
        face_img = frame[y1:y2, x1:x2]

        if face_img.size != 0:
            # Preprocess face
            face_tensor = torch.tensor(face_img.transpose((2,0,1)), dtype=torch.float32)
            face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0), size=(160,160))
            face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
            with torch.no_grad():
                embedding = resnet(face_tensor).squeeze().numpy()
            print("Face captured!")
            break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if embedding is None:
    print("No face detected. Exiting.")
    conn.close()
    sys.exit(1)

# -----------------------------
# Check if person already exists by embedding
# -----------------------------
c.execute("SELECT name, embedding FROM users")
rows = c.fetchall()
threshold = 0.6  # cosine similarity threshold

for db_name, db_emb_bytes in rows:
    db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
    sim = cosine_similarity(embedding, db_emb)
    if sim > threshold:
        print(f"Face already exists in the database as '{db_name}' (similarity={sim:.2f}). Camera off.")
        conn.close()
        sys.exit(0)
    elif name == db_name:
        print(f"With This {name} face is alredy Exist")
        sys.exit(0)

# -----------------------------
# Store embedding in DB
# -----------------------------
embedding_bytes = embedding.astype(np.float32).tobytes()
c.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, embedding_bytes))
conn.commit()
conn.close()
print(f"{name} has been registered successfully.")
