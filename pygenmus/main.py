import os
import random
import json
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message

# Define instruments, notes, and styles
instruments = ["piano", "guitar", "violin", "drums"]
notes = ["C", "D", "E", "F", "G", "A", "B"]
styles = ["fingerpicking", "strumming", "legato", "staccato"]

# Define the root directory for the dataset
root_dir = "Dataset Root"

# Create the root directory if it does not exist
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# Create a metadata dictionary to store the information of each WAV file
metadata = {}

# Initialize a file counter
file_counter = 0

# Initialize a FluidSynth instance for MIDI rendering
fs = MidiFluidSynth()

# Loop through each instrument
for instrument in instruments:
    # Create a subdirectory for the instrument
    instrument_dir = os.path.join(root_dir, instrument)
    if not os.path.exists(instrument_dir):
        os.mkdir(instrument_dir)

    # Loop through each note
    for note in notes:
        # Create a subdirectory for the note
        note_dir = os.path.join(instrument_dir, note)
        if not os.path.exists(note_dir):
            os.mkdir(note_dir)

        # Loop through each style
        for style in styles:
            # Generate a file name with the format instrument_note_style.wav using the file counter
            file_name = f"{instrument}_{note}_{style}.wav"
            file_path = os.path.join(note_dir, file_name)

            # Increment the file counter
            file_counter += 1

            # Generate a new MIDI file with the specified instrument, note, and style
            midi_file = MidiFile()
            midi_track = MidiTrack()
            midi_file.tracks.append(midi_track)

            # Set the program change message to select the instrument
            program = instruments.index(instrument) + 1  # The program number is 1-based
            midi_track.append(Message("program_change", program=program))

            # Set the note on and note off messages to play the note with a fixed velocity and duration
            note_value = notes.index(note) + 60  # The note value is 60-based (C4 is 60)
            velocity = 100  # The velocity is fixed at 100
            duration = 1000  # The duration is fixed at 1000 milliseconds

            # If the instrument is guitar, add some messages to change the playing style (e.g., fingerpicking or strumming)
            if instrument == "guitar":
                # If the style is fingerpicking, use a different channel for each string and play them separately with different velocities and timings
                if style == "fingerpicking":
                    # Define the channels and strings for fingerpicking (channel 0 is reserved for drums)
                    channels = [1, 2, 3, 4, 5, 6]
                    strings = [40, 45, 50, 55, 59, 64]

                    # Loop through each channel and string
                    for channel, string in zip(channels, strings):
                        # Set the program change message to select the guitar instrument on each channel
                        midi_track.append(Message("program_change", channel=channel, program=program))

                        # Calculate the note value by adding the string value and the note index
                        note_value = string + notes.index(note)

                        # Generate a random velocity between 64 and 127 for each string
                        velocity = random.randint(64, 127)

                        # Generate a random timing between 0 and 500 milliseconds for each string
                        timing = random.randint(0, 500)

                        # Add the note on and note off messages for each string with the calculated values
                        midi_track.append(Message("note_on", channel=channel, note=note_value, velocity=velocity))
                        midi_track.append(Message("note_off", channel=channel, note=note_value, time=duration + timing))

            # Save the MIDI file to a temporary location
            temp_midi_path = f"temp_{file_name}.mid"
            midi_file.save(temp_midi_path)

            # Convert the MIDI file to a WAV file using the FluidSynth instance
            fs.midi_to_audio(temp_midi_path, file_path)

            # Delete the temporary MIDI file
            os.remove(temp_midi_path)

            # Add the file information to the metadata dictionary
            metadata[file_name] = {"instrument": instrument, "note": note, "style": style}

# Write the metadata dictionary to a JSON file in the root directory
metadata_path = os.path.join(root_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)