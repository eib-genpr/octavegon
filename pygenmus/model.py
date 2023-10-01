import os
import json
import numpy as np
import tensorflow as tf
import librosa
from keras.layers import Dense, LSTM, Input, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

notes = ["C", "D", "E", "F", "G", "A", "B"] 

metadata_path = "Dataset Root/metadata.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)

segment_length = 44100 * 3
sample_rate = 44100
input_shape = (segment_length, 1) 

model = Sequential()
model.add(Input(shape=input_shape))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(len(notes), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def load_audio_data(file_path, segment_length, sample_rate):
    audio_data, _ = librosa.load(file_path, sr=sample_rate)
    num_segments = len(audio_data) // segment_length
    segments = []

    for i in range(num_segments):
        segment = audio_data[i * segment_length : (i + 1) * segment_length]
        segments.append(segment.reshape(-1, 1)) 

    return segments

def transcribe_segment(segment):
    segment = segment.reshape((1, segment_length, 1))
    note_prediction = model.predict(segment)
    note_index = np.argmax(note_prediction)
    note = notes[note_index]

    return note

def train_model(metadata, segment_length, sample_rate, batch_size, num_epochs):
    keys = list(metadata.keys())
    np.random.shuffle(keys)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True) 
    ]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for key in keys:
            file_name = key
            file_path = os.path.join("Dataset Root", metadata[file_name]["instrument"], metadata[file_name]["note"], file_name)
            segments = load_audio_data(file_path, segment_length, sample_rate)
            note_encoded = to_categorical(notes.index(metadata[file_name]["note"]), num_classes=len(notes))
            note_batch = np.array([note_encoded] * len(segments))

            history = model.fit(
                np.array(segments),
                note_batch,
                batch_size=batch_size,
                epochs=1,
                callbacks=callbacks,
                verbose=1
            )

            print(f"File: {file_name}, Loss: {history.history['loss'][0]}, Accuracy: {history.history['accuracy'][0]}")

batch_size = 32
num_epochs = 50

train_model(metadata, segment_length, sample_rate, batch_size, num_epochs)

model.save("music_transcription_model.h5")