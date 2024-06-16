import cv2
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while count < 100:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f'dataset/user{count}.jpg', face)
        cv2.imshow('Face', face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

