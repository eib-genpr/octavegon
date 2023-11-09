import os
import json
import wave

def remove_entry_from_metadata(metadata_path, filename_to_remove):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    base_filename = os.path.basename(filename_to_remove)
    if base_filename in metadata:
        del metadata[base_filename]
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def get_audio_duration(audio_path):
    with wave.open(audio_path, 'rb') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
        return duration

dataset_root = ".\\dataset_root"
directories = ['test', 'train']

for directory in directories:
    audio_directory = os.path.join(dataset_root, directory)
    metadata_file = os.path.join(dataset_root, f"{directory}_metadata.json")

    for filename in os.listdir(audio_directory):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_directory, filename)
            duration = get_audio_duration(audio_path)

            if not 6.99 <= duration <= 7.01:
                os.remove(audio_path)
                remove_entry_from_metadata(metadata_file, filename)
                print(f"Deleted from {directory}: {filename}")

print("Audio files processed and metadata updated.")
