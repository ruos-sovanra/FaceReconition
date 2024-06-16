import cv2
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and label encoder
model = load_model('face_recognition_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        predictions = model.predict(preprocessed_face)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
        
        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

