import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_and_preprocess_audio(file_path, max_duration=5.0, sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=max_duration)
        # Convert audio signal to spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def prepare_data(file_paths, labels, max_duration=5.0):
    X = []
    y = []
    for file_path, label in zip(file_paths, labels):
        spectrogram = load_and_preprocess_audio(file_path, max_duration)
        if spectrogram is not None:
            X.append(spectrogram)
            y.append(label)
    return np.array(X), to_categorical(y)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

def main():
    # Paths to your dataset
    audio_file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav']  # Add your paths here
    labels = [0, 1]  # Corresponding labels for audio files
    
    # Prepare data
    X, y = prepare_data(audio_file_paths, labels)

    # Add channel dimension
    X = X[..., np.newaxis]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
