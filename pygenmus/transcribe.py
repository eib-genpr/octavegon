import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf

model = tf.keras.models.load_model("music_transcription_model.h5")
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

def transcribe(audio_segment):
    segment = audio_segment.reshape((1, segment_length, 1))

    note_prediction = model.predict(segment)

    note_index = np.argmax(note_prediction)

    midi_value = note_to_midi.get(note, 0) 

    return note, midi_value

input_audio_file = "input_audio.mp3" 
output_dir = "output_segments" 

os.makedirs(output_dir, exist_ok=True)

segment_length = 3 * 44100 

audio_data, sample_rate = librosa.load(input_audio_file, sr=None)

all_notes = []
all_midi_values = []

for i, start_sample in enumerate(range(0, len(audio_data), segment_length)):
    segment = audio_data[start_sample:start_sample + segment_length]
    note, midi_value = transcribe(segment)

    all_notes.append(note)
    all_midi_values.append(midi_value)

    print(f"Segment {i+1}: Note {note}, MIDI Value {midi_value}")

output_file = os.path.join(output_dir, "output_piano.wav")
sf.write(output_file, np.zeros(1), sample_rate)  

for note, midi_value in zip(all_notes, all_midi_values):
    duration = 3 
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequency = 440.0 * (2.0 ** ((midi_value - 69) / 12.0))
    sine_wave = np.sin(2 * np.pi * frequency * t)

    with open(output_file, "ab") as f:
        np.savetxt(f, sine_wave)

print(f"Output piano composition saved to {output_file}")
