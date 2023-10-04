import json
import os
import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.layers import Dense, LSTM, Input, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, TensorBoard

# Define constants
NOTES = ["C", "D", "E", "F", "G", "A", "B"]
SEGMENT_LENGTH = 44100 * 3  # Adjust as needed
SAMPLE_RATE = 44100
INPUT_SHAPE = (SEGMENT_LENGTH, 1)
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_NOTES = len(NOTES)
NUM_INSTRUMENTS = 2


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
                segment = pitch_shift(segment, sample_rate)

            segments.append(segment.reshape(-1, 1))

        return segments
    except Exception as e:
        print(f"Error loading audio data: {e}")
        return []


def train_one_epoch(metadata, segment_length, sample_rate, model, epoch, log_file):
    keys = list(metadata.keys())
    np.random.shuffle(keys)

    losses = []
    accuracies = []
    f1_scores = []

    for i, key in enumerate(keys):
        file_name = key
        file_path = os.path.join(
            "dataset_root", metadata[file_name].get("instrument1", ""),
            metadata[file_name].get("note1", ""), file_name)

        # Data augmentation for a portion of the data
        augment = np.random.rand() < 0.5
        segments = load_audio_data(
            file_path, segment_length, sample_rate, augment=augment)

        if "instrument2" in metadata[file_name] and "note2" in metadata[file_name]:
            notes_encoded = [to_categorical(NOTES.index(metadata[file_name]["note1"]), num_classes=NUM_NOTES),
                             to_categorical(NOTES.index(metadata[file_name]["note2"]), num_classes=NUM_NOTES)]
            notes_batch = np.array([notes_encoded] * len(segments))
        else:
            notes_encoded = [to_categorical(NOTES.index(
                metadata[file_name]["note1"]), num_classes=NUM_NOTES)]
            notes_batch = np.array([notes_encoded] * len(segments))

        if len(segments) > 0:
            # Mini-batch training loop
            num_batches = len(segments) // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                batch_segments = np.array(segments[start_idx:end_idx])
                batch_labels = notes_batch[start_idx:end_idx]

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

                    # F1-score as an additional metric
                    predicted_labels = np.argmax(
                        model.predict(batch_segments), axis=1)
                    true_labels = np.argmax(batch_labels, axis=2)
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
    model.add(Dense(NUM_NOTES * NUM_INSTRUMENTS, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training.log")
    log_file = open(log_file_path, "w")

    with open("dataset_root/metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        losses, accuracies, f1_scores = train_one_epoch(
            metadata, SEGMENT_LENGTH, SAMPLE_RATE, model, epoch, log_file)

    log_file.close()
    model.save("generated_models/music_transcription_model.h5")


if __name__ == "__main__":
    main()
