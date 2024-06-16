from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('models/face_recognition_model.h5')

images = np.load('images.npy')
labels = np.load('labels.npy')

X_train, X_test, y_train, y_test = train_test_split(images, labels_enc, test_size=0.2, random_state=42)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

