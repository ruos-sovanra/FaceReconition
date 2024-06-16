from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = load_model('models/face_recognition_model.h5')

# Load preprocessed data
images = np.load('images.npy')
labels = np.load('labels.npy')

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')
labels_enc = label_encoder.transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_enc, test_size=0.2, random_state=42)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')

# Assuming your model outputs categorical probabilities (softmax), 
# and y_test is one-hot encoded or categorical labels
# Example:
# y_test_pred = model.predict(X_test)
# predicted_labels = np.argmax(y_test_pred, axis=1)

# If your model predicts probabilities, you can use `categorical_crossentropy` for training
# loss = keras.losses.categorical_crossentropy(y_test, y_test_pred)

# Plotting training history if available (assuming you have trained the model previously)
# Replace 'history' with your actual training history object from model.fit(...)
# Example:
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# Then plot the history:
# plt.plot(history.history['accuracy'], label='train_accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')

# Placeholder plot (comment out once you have actual history)
plt.plot([0, 1, 2], [0, 1, 0], label='train_accuracy')  # Example placeholder data
plt.plot([0, 1, 2], [0, 0.5, 1], label='val_accuracy')  # Example placeholder data

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

