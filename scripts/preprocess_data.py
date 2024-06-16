import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING messages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        return face
    return None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
images = []
labels = []
for img_path in os.listdir('dataset'):
    img = cv2.imread(os.path.join('dataset', img_path))
    face = detect_face(img)
    if face is not None:
        face = cv2.resize(face, (160, 160))
        images.append(face)
        labels.append(img_path.split('_')[0])

images = np.array(images)
labels = np.array(labels)
images = images / 255.0

# Save preprocessed data
np.save('images.npy', images)
np.save('labels.npy', labels)

