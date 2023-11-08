import os
import json
import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.layers import Dense, LSTM, Input, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, TensorBoard

NOTES = ["C", "D", "E", "F", "G", "A", "B"]
SEGMENT_LENGTH = 44100 * 7
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
    print(f"Loading data from: {file_path}")
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


def train_one_epoch(metadata, segment_length, sample_rate, model, epoch, log_file, tensorboard_callback):
    keys = list(metadata.keys())
    np.random.shuffle(keys)

    losses = []
    accuracies = []
    f1_scores = []

    for i, key in enumerate(keys):
        file_name = key
        file_path = os.path.join("dataset_root", file_name)

        augment = np.random.rand() < 0.5
        segments = load_audio_data(
            file_path, segment_length, sample_rate, augment=augment)

        instrument_data = metadata[file_name]
        for instrument_info in instrument_data:
            note_encoded = to_categorical(NOTES.index(
                instrument_info['note']), num_classes=NUM_NOTES)
            notes_batch = np.array([note_encoded] * len(segments))

            if len(segments) > 0:
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
                        callbacks=[tensorboard_callback]
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

                        log_str = f"Epoch: {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss}, Accuracy: {accuracy}, F1-Score: {f1}\n"
                        print(log_str)
                        log_file.write(log_str)

    return losses, accuracies, f1_scores


def main():
    print("Starting main function...")
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Flatten())
    model.add(Dense(NUM_NOTES, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(
        filepath='best_model.h5', save_best_only=True, verbose=1)

    with open('dataset_root/metadata.json', 'r') as f:
        metadata = json.load(f)

    log_file_path = "/logs/training_log.txt"
    with open(log_file_path, "a") as log_file:
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(metadata, SEGMENT_LENGTH, SAMPLE_RATE,
                            model, epoch, log_file, tensorboard_callback)

    model_file = "music_transcription_model.h5"
    model.save(model_file)


if __name__ == "__main__":
    main()
