import json
import os
import librosa
from keras.layers import Dense, LSTM, Input, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
import logging

NOTES = ["C", "D", "E", "F", "G", "A", "B"]
SEGMENT_LENGTH = 44100 * 3
SAMPLE_RATE = 44100
INPUT_SHAPE = (SEGMENT_LENGTH, 1)
BATCH_SIZE = 32
NUM_EPOCHS = 50


def pitch_shift(segment, sample_rate, max_steps=2.0):
    steps = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(segment, sr=sample_rate, n_steps=steps)


def load_audio_data(file_path, segment_length, sample_rate, augment=False):
    try:
        audio_data, _ = librosa.load(file_path, sr=sample_rate)
        num_segments = len(audio_data) // segment_length
        segments = []
        for i in range(num_segments):
            segment = audio_data[i * segment_length: (i + 1) * segment_length]
            if augment and np.random.rand() < 0.5:
                # //TODO Apply data augmentation techniques here
                # Random pitch shift
                segment = pitch_shift(segment, sample_rate)
            segments.append(segment.reshape(-1, 1))
        return segments
    except Exception as e:
        print(f"Error loading audio data: {e}")
        return []


def transcribe_segment(segment, model):
    segment = segment.reshape((1, SEGMENT_LENGTH, 1))
    note_prediction = model.predict(segment)
    note_index = np.argmax(note_prediction)
    note = NOTES[note_index]
    return note


def train_one_epoch(metadata, segment_length, sample_rate, model, epoch, log_file):
    keys = list(metadata.keys())
    np.random.shuffle(keys)

    losses = []
    accuracies = []
    f1_scores = []

    for i, key in enumerate(keys):
        file_name = key
        file_path = os.path.join(
            "Dataset Root", metadata[file_name]["instrument"], metadata[file_name]["note"], file_name)

        # Enable data augmentation for a portion of the data
        augment = np.random.rand() < 0.5
        segments = load_audio_data(
            file_path, segment_length, sample_rate, augment=augment)
        note_encoded = to_categorical(NOTES.index(
            metadata[file_name]["note"]), num_classes=len(NOTES))
        note_batch = np.array([note_encoded] * len(segments))

        if len(segments) > 0:
            # Mini-batch training loop
            num_batches = len(segments) // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                batch_segments = np.array(segments[start_idx:end_idx])
                batch_labels = note_batch[start_idx:end_idx]

                history = model.fit(
                    batch_segments,
                    batch_labels,
                    batch_size=BATCH_SIZE,
                    epochs=1,
                    verbose=0,
                )
                loss = history.history['loss'][0]
                accuracy = history.history['accuracy'][0]

                if not np.isnan(loss) and not np.isnan(accuracy):
                    losses.append(loss)
                    accuracies.append(accuracy)

                    predicted_labels = np.argmax(
                        model.predict(batch_segments), axis=1)
                    true_labels = np.argmax(batch_labels, axis=1)
                    f1 = f1_score(
                        true_labels, predicted_labels, average='weighted')
                    f1_scores.append(f1)

                    log_str = f"Epoch: {epoch + 1}, Entry {i + 1}/{len(keys)}, File: {file_name}, Loss: {loss}, Accuracy: {accuracy}, F1-Score: {f1}\n"
                    print(log_str)
                    log_file.write(log_str)

    return losses, accuracies, f1_scores


def main():
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Flatten())
    model.add(Dense(len(NOTES), activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Configure logging
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training.log")
    log_file = open(log_file_path, "w")

    # Load metadata
    with open("Dataset Root\metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        losses, accuracies, f1_scores = train_one_epoch(
            metadata, SEGMENT_LENGTH, SAMPLE_RATE, model, epoch, log_file)

    log_file.close()
    model.save("music_transcription_model.h5")


if __name__ == "__main__":
    main()
