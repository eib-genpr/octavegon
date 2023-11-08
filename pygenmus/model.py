import os
import json
import librosa
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Input, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, TensorBoard

ROOT_DIR = "dataset_root"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
TRAIN_METADATA = os.path.join(ROOT_DIR, "train_metadata.json")
TEST_METADATA = os.path.join(ROOT_DIR, "test_metadata.json")
NOTES = ["C", "D", "E", "F", "G", "A", "B"]
SEGMENT_LENGTH = 44100 * 7
SAMPLE_RATE = 44100
INPUT_SHAPE = (SEGMENT_LENGTH, 1)
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_NOTES = len(NOTES)

def load_audio_data(file_path, sample_rate):
    try:
        audio_data, _ = librosa.load(file_path, sr=sample_rate)
        return audio_data.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading audio data: {e}")
        return None

def load_dataset(directory, metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    dataset = []
    for filename, notes in metadata.items():
        file_path = os.path.join(directory, filename)
        audio_data = load_audio_data(file_path, SAMPLE_RATE)
        if audio_data is None:
            continue
        for note_info in notes:
            note_encoded = to_categorical(NOTES.index(note_info['note']), num_classes=NUM_NOTES)
            dataset.append((audio_data, note_encoded))
    
    return dataset

def main():
    print("Loading data...")
    train_data = load_dataset(TRAIN_DIR, TRAIN_METADATA)
    test_data = load_dataset(TEST_DIR, TEST_METADATA)

    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_NOTES, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Starting training...")
    train_segments, train_labels = zip(*train_data)
    train_segments = np.array(train_segments)
    train_labels = np.array(train_labels)

    model.fit(train_segments, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    print("Evaluating model...")
    test_segments, test_labels = zip(*test_data)
    test_segments = np.array(test_segments)
    test_labels = np.array(test_labels)

    _, accuracy = model.evaluate(test_segments, test_labels)
    print(f"Test accuracy: {accuracy}")

    print("Training complete.")

if __name__ == "__main__":
    main()
