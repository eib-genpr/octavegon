import os
import random
import json
import numpy as np
import fluidsynth
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message

instruments = ["piano", "guitar", "violin", "drums"]
notes = ["C", "D", "E", "F", "G", "A", "B"]
styles = ["fingerpicking", "strumming", "legato", "staccato"]

root_dir = "Dataset Root"

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

metadata = {}

file_counter = 0

fs = MidiFluidSynth()

for instrument in instruments:
    instrument_dir = os.path.join(root_dir, instrument)
    if not os.path.exists(instrument_dir):
        os.mkdir(instrument_dir)

    for note in notes:
        note_dir = os.path.join(instrument_dir, note)
        if not os.path.exists(note_dir):
            os.mkdir(note_dir)

        for style in styles:
            file_name = f"{instrument}_{note}_{style}.wav"
            file_path = os.path.join(note_dir, file_name)

            file_counter += 1

            midi_file = MidiFile()
            midi_track = MidiTrack()
            midi_file.tracks.append(midi_track)

            program = instruments.index(instrument) + 1
            midi_track.append(Message("program_change", program=program))

            note_value = notes.index(note) + 60
            velocity = 100
            duration = 1000

            if instrument == "guitar":
                if style == "fingerpicking":
                    channels = [1, 2, 3, 4, 5, 6]
                    strings = [40, 45, 50, 55, 59, 64]

                    for channel, string in zip(channels, strings):
                        midi_track.append(Message("program_change", channel=channel, program=program))
                        note_value = string + notes.index(note)
                        velocity = random.randint(64, 127)
                        timing = random.randint(0, 500)
                        midi_track.append(Message("note_on", channel=channel, note=note_value, velocity=velocity))
                        midi_track.append(Message("note_off", channel=channel, note=note_value, time=duration + timing))

            temp_midi_path = f"temp_{file_name}.mid"
            midi_file.save(temp_midi_path)

            fs.midi_to_audio(temp_midi_path, file_path)
            os.remove(temp_midi_path)

            metadata[file_name] = {"instrument": instrument, "note": note, "style": style}

metadata_path = os.path.join(root_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
