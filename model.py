import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np

# Define a CNN model for audio classification
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # Output: 2 classes (Cardio or Pulmonary)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(X_train, y_train, input_shape):
    model = create_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# Feature extraction from audio
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

