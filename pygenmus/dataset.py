import os
import json
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message

instruments = ["piano", "guitar", "violin", "drums", "flute", "trumpet", "saxophone", "clarinet", "synthesizer", "electric_guitar", "bass_guitar"]
notes = ["C", "D", "E", "F", "G", "A", "B"]

#Add more instruments and styles
instrument_programs = {
    "piano": 0,
    "guitar": 24,
    "violin": 40,
    "drums": 118,
    "flute": 73,
    "trumpet": 56,
    "saxophone": 66,
    "clarinet": 71,
    "synthesizer": 81,
    "electric_guitar": 27,
    "bass_guitar": 33,
}

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

        file_name = f"{instrument}_{note}.wav"
        file_path = os.path.join(note_dir, file_name)

        file_counter += 1

        
        midi_file = MidiFile()
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)

        program = instrument_programs.get(instrument, 0)  
        midi_track.append(Message("program_change", program=program))

        note_value = notes.index(note) + 60
        velocity = 100
        duration = 1000

        midi_track.append(Message("note_on", note=note_value, velocity=velocity))
        midi_track.append(Message("note_off", note=note_value, time=duration))

        temp_midi_path = f"temp_{file_name}.mid"
        midi_file.save(temp_midi_path)

        fs.midi_to_audio(temp_midi_path, file_path)
        os.remove(temp_midi_path)

        metadata[file_name] = {"instrument": instrument, "note": note}

metadata_path = os.path.join(root_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
