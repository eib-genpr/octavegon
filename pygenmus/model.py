import os
import json
import numpy as np
import tensorflow as tf
import librosa
from keras.layers import Dense, LSTM, Input, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime

NOTES = ["C", "D", "E", "F", "G", "A", "B"]
SEGMENT_LENGTH = 44100 * 3
SAMPLE_RATE = 44100
INPUT_SHAPE = (SEGMENT_LENGTH, 1)
BATCH_SIZE = 32
NUM_EPOCHS = 50

def load_audio_data(file_path, segment_length, sample_rate):
    try:
        audio_data, _ = librosa.load(file_path, sr=sample_rate)
        num_segments = len(audio_data) // segment_length
        segments = []

        for i in range(num_segments):
            segment = audio_data[i * segment_length : (i + 1) * segment_length]
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

def train_one_epoch(metadata, segment_length, sample_rate, model, epoch):
    keys = list(metadata.keys())
    np.random.shuffle(keys)

    for i, key in enumerate(keys):
        file_name = key
        file_path = os.path.join("Dataset Root", metadata[file_name]["instrument"], metadata[file_name]["note"], file_name)
        segments = load_audio_data(file_path, segment_length, sample_rate)
        note_encoded = to_categorical(NOTES.index(metadata[file_name]["note"]), num_classes=len(NOTES))
        note_batch = np.array([note_encoded] * len(segments))

        if len(segments) > 0:
            history = model.fit(
                np.array(segments),
                note_batch,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=1
            )

            print(f"Epoch: {epoch + 1}, Entry {i + 1}/{len(keys)}, File: {file_name}, Loss: {history.history['loss'][0]}, Accuracy: {history.history['accuracy'][0]}")

model = Sequential()
model.add(Input(shape=INPUT_SHAPE))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(len(NOTES), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    gpu_device = physical_devices[0]
    print("GPU Name: ", gpu_device.name)
    print("GPU Memory Limit: ", tf.config.experimental.get_memory_info(gpu_device).limit)

metadata_path = "Dataset Root/metadata.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_one_epoch(metadata, SEGMENT_LENGTH, SAMPLE_RATE, model, epoch)

model.save("music_transcription_model.h5")
