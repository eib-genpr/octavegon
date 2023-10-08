import os
import librosa
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("generated_models/music_transcription_model.h5")
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
    segment = audio_segment.reshape((1, segment_length, 1))
    note_prediction = model.predict(segment)

    note_index = np.argmax(note_prediction)
    note = notes[note_index]

    return note

input_audio_file = r"input/fur_elise.mp3"  # Adjust the input file path
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# Adjust segment_length according to your needs
segment_length = 3 * 44100  # 3 seconds

audio_data, sample_rate = librosa.load(input_audio_file, sr=None)

all_notes = []

for i, start_sample in enumerate(range(0, len(audio_data), segment_length)):
    segment = audio_data[start_sample:start_sample + segment_length]
    note = transcribe(segment, model)

    all_notes.append(note)

    print(f"Segment {i+1}: Note {note}")

output_notes_file = r"output/transcribed_notes.txt"
with open(output_notes_file, "w") as f:
    for note in all_notes:
        f.write(note + "\n")

print(f"Transcribed notes saved to {output_notes_file}")
