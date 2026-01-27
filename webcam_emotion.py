import cv2
import torch
import numpy as np
from emotion_model import Emotiondetection

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
checkpoint = torch.load("emotion_model.pth", map_location=device)

model = Emotiondetection().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

classes = checkpoint["classes"]

# face detector
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 117, 24), 2)

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # SAME preprocessing as training
        face = face / 255.0
        face = (face - 0.5) / 0.5

        face = torch.tensor(face, dtype=torch.float32)
        face = face.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)
            pred = torch.argmax(output, dim=1).item()
            emotion = classes[pred]

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
