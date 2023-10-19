import os
import librosa
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model(
    "generated_models/music_transcription_model.h5")
notes = ["C", "D", "E", "F", "G", "A", "B"]

note_to_midi = {
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "A": 69,
    "B": 71,
}


def transcribe(audio_segment, model):
    segment_length = len(audio_segment)  # Get the actual segment length

    # Check if the segment length matches the model's input shape
    model_segment_length = model.input_shape[1]
    if len(audio_segment) != model_segment_length:
        if len(audio_segment) < model_segment_length:
            # Pad the segment with zeros to match the model's input length
            audio_segment = np.pad(
                audio_segment, (0, model_segment_length - len(audio_segment)), 'constant')
        else:
            # Trim the segment to match the model's input length
            audio_segment = audio_segment[:model_segment_length]

    segment = audio_segment.reshape((1, model_segment_length, 1))
    note_prediction = model.predict(segment)

    note_indices = np.argmax(note_prediction, axis=2)[0]  # Fix axis argument
    notes = [notes[i] for i in note_indices]

    return notes


input_audio_file = r"input/fur_elise.mp3"  # Adjust the input file path
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# Adjust segment_length according to your needs
segment_length = 3 * 44100  # 3 seconds

audio_data, sample_rate = librosa.load(input_audio_file, sr=None)

all_notes = []

# Load the metadata file
with open("dataset_root/metadata.json", "r") as metadata_file:
    metadata = json.load(metadata_file)

for i, start_sample in enumerate(range(0, len(audio_data), segment_length)):
    segment = audio_data[start_sample:start_sample + segment_length]

    # Adjust the segment length to match the model's input shape
    if len(segment) % model.input_shape[1] != 0:
        segment = np.pad(
            segment, (0, model.input_shape[1] - (len(segment) % model.input_shape[1])), 'constant')

    notes = transcribe(segment, model)

    all_notes.extend(notes)

    print(f"Segment {i+1}: Notes {notes}")

output_notes_file = r"output/transcribed_notes.txt"
with open(output_notes_file, "w") as f:
    for note in all_notes:
        f.write(note + "\n")

print(f"Transcribed notes saved to {output_notes_file}")
