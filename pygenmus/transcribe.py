import os
import librosa
import numpy as np
import tensorflow as tf

# Load your pre-trained model
model = tf.keras.models.load_model("music_transcription_model.h5")
notes = ["C", "D", "E", "F", "G", "A", "B"] 


# Define a function to transcribe notes for a 3-second audio segment
def transcribe(audio_segment):
    # Ensure the segment is in the shape expected by your model
    segment = audio_segment.reshape((1, segment_length, 1))

    # Use your model to predict the note for the segment
    note_prediction = model.predict(segment)

    # Decode the prediction to get the note (e.g., using `np.argmax`)
    note_index = np.argmax(note_prediction)
    note = notes[note_index]  # Get the corresponding note from your notes list

    return note

# Input audio file path (mp3 or wav)
input_audio_file = "input_audio.mp3"  # Update with your file path
output_dir = "output_segments"  # Directory to save the output segments

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define segment length (3 seconds)
segment_length = 3 * 44100  # Adjust as needed based on your sample rate

# Load the input audio file and convert to wav if needed
if input_audio_file.endswith(".mp3"):
    # You can add code here to convert mp3 to wav if needed
    pass

# Load the audio data
audio_data, sample_rate = librosa.load(input_audio_file, sr=None)

# Split the audio into 3-second segments and transcribe the notes
for i, start_sample in enumerate(range(0, len(audio_data), segment_length)):
    segment = audio_data[start_sample:start_sample + segment_length]
    note = transcribe(segment)

    print(f"Segment {i+1}: Note {note}")