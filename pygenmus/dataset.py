import os
import json
import soundfile as sf
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message
import librosa
import numpy as np
import random

INSTRUMENTS = ["piano", "guitar", "violin", "flute", "trumpet",
               "saxophone", "clarinet", "synthesizer", "electric_guitar", "bass_guitar"]
NOTES = ["C", "D", "E", "F", "G", "A", "B"]

INSTRUMENT_PROGRAMS = {
    "piano": 0,
    "guitar": 24,
    "violin": 40,
    "flute": 73,
    "trumpet": 56,
    "saxophone": 66,
    "clarinet": 71,
    "synthesizer": 81,
    "electric_guitar": 27,
    "bass_guitar": 33,
}

ROOT_DIR = "dataset_root"
AUGMENT = True
OVERLAP = True
SEGMENT_LENGTH = 44100 * 7
SAMPLE_RATE = 44100
VOLUME_FACTOR = 10.0

def random_duration():
    return round(random.uniform(0.1, 2.0), 3)

def handle_overlapping_notes(active_notes, end_timestamp):
    notes_to_remove = []
    for note, (start_timestamp, duration) in active_notes.items():
        if start_timestamp + duration < end_timestamp:
            notes_to_remove.append(note)
    for note in notes_to_remove:
        del active_notes[note]

def generate_segment():
    midi_file = MidiFile()
    midi_track = MidiTrack()
    midi_file.tracks.append(midi_track)

    total_duration = 0.0
    active_notes = {}

    while total_duration < 7.0:
        instrument = random.choice(INSTRUMENTS)
        note = random.choice(NOTES)
        duration = random_duration()

        program = INSTRUMENT_PROGRAMS.get(instrument, 0)
        midi_track.append(
            Message("program_change", program=program))

        note_value = NOTES.index(note) + 60
        velocity = 100
        duration_ms = int(duration * 1000)

        start_timestamp = total_duration
        end_timestamp = total_duration + duration
        handle_overlapping_notes(active_notes, start_timestamp)

        midi_track.append(
            Message("note_on", note=note_value, velocity=velocity))
        midi_track.append(
            Message("note_off", note=note_value, time=duration_ms))

        active_notes[note_value] = (start_timestamp, duration)
        total_duration = end_timestamp

    return midi_file

def combine_segments(segment1, segment2):
    combined_segment = MidiFile()
    combined_segment.tracks.append(segment1.tracks[0])
    combined_segment.tracks.append(segment2.tracks[0])

    return combined_segment

def main():
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    metadata = {}
    fs = MidiFluidSynth()

    for i in range(100):
        segment1 = generate_segment()
        segment2 = generate_segment()

        combined_segment = combine_segments(segment1, segment2)

        combined_segment_name = f"combined_segment_{i}.wav"
        combined_file_path = os.path.join(ROOT_DIR, combined_segment_name)

        total_duration = 7.0

        temp_midi_path = os.path.join(ROOT_DIR, f"temp_{combined_segment_name}.mid")
        combined_segment.save(temp_midi_path)

        fs.midi_to_audio(temp_midi_path, combined_file_path)
        os.remove(temp_midi_path)

        instrument_metadata = {}
        program = 0
        for track in combined_segment.tracks:
            active_notes = {}
            for msg in track:
                if msg.type == "program_change":
                    program = msg.program
                if msg.type == "note_on":
                    note = msg.note
                    if note in active_notes:
                        start_timestamp = active_notes[note][0]
                        duration = (msg.time / 1000.0) - start_timestamp
                        instrument_metadata.setdefault(
                            INSTRUMENTS[program % len(INSTRUMENTS)], {}).setdefault(
                            NOTES[note % 12], []).append({
                            "duration": round(duration, 3),
                            "start_timestamp": round(start_timestamp, 3),
                            "end_timestamp": round(start_timestamp + duration, 3)
                        })
                        del active_notes[note]
                    else:
                        active_notes[note] = (msg.time / 1000.0, 0.0)
                elif msg.type == "note_off":
                    note = msg.note
                    if note in active_notes:
                        start_timestamp = active_notes[note][0]
                        duration = active_notes[note][1]
                        instrument_metadata.setdefault(
                            INSTRUMENTS[program % len(INSTRUMENTS)], {}).setdefault(
                            NOTES[note % 12], []).append({
                            "duration": round(duration, 3),
                            "start_timestamp": round(start_timestamp, 3),
                            "end_timestamp": round(start_timestamp + duration, 3)
                        })
                        del active_notes[note]

        metadata[combined_segment_name] = instrument_metadata

    metadata_path = os.path.join(ROOT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()
