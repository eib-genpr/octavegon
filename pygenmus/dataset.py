import os
import json
import soundfile as sf
from midi2audio import FluidSynth
from mido import MidiFile, MidiTrack, Message, MetaMessage
import random
import shutil
import numpy as np


INSTRUMENTS = ["piano", "guitar", "violin", "flute", "trumpet",
               "saxophone", "clarinet", "synthesizer", "electric_guitar", "bass_guitar"]
NOTES = ["C", "D", "E", "F", "G", "A", "B"]

INSTRUMENT_PROGRAMS = {
    "piano": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "guitar": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    "violin": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    "flute": [73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
    "trumpet": [56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
    "saxophone": [66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
    "clarinet": [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    "synthesizer": [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    "electric_guitar": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    "bass_guitar": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
}

ROOT_DIR = "dataset_root"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
SEGMENT_DURATION = 3.0
SAMPLE_RATE = 44100
SOUNDFONT_PATH = "C:\\soundfonts\\FluidR3_GM.sf2"
TRAIN_SPLIT = 0.8


def random_duration(total_duration, segment_duration):
    max_duration = segment_duration - total_duration
    return round(random.uniform(0.1, min(2.0, max_duration)), 3)


def generate_segment():
    midi_file = MidiFile()
    midi_track = MidiTrack()
    midi_file.tracks.append(midi_track)

    total_duration = 0.0
    all_notes = []

    while total_duration < SEGMENT_DURATION:
        instrument = random.choice(INSTRUMENTS)
        note = random.choice(NOTES)
        duration = random_duration(
            total_duration=total_duration, segment_duration=SEGMENT_DURATION)
        note_value = NOTES.index(note) + 60
        velocity = random.randint(80, 120)

        duration_ticks = int(midi_file.ticks_per_beat * duration)

        programs = INSTRUMENT_PROGRAMS[instrument]
        program = random.choice(programs)

        start_timestamp = total_duration
        end_timestamp = start_timestamp + duration

        if end_timestamp > SEGMENT_DURATION:
            end_timestamp = SEGMENT_DURATION
            duration = end_timestamp - start_timestamp
            duration_ticks = int(midi_file.ticks_per_beat * duration)

        midi_track.append(Message("program_change", program=program, time=0))
        midi_track.append(Message("note_on", note=note_value,
                          velocity=velocity, time=0))
        midi_track.append(Message("note_off", note=note_value,
                          velocity=0, time=duration_ticks))

        all_notes.append({
            "instrument": instrument,
            "note": note,
            "start_time": start_timestamp,
            "end_time": end_timestamp,
            "duration": duration,
            "program": program
        })

        total_duration = end_timestamp

    all_notes.sort(key=lambda x: x['start_time'])

    return midi_file, all_notes


def combine_segments(segment1, segment2, ticks_per_beat=480):
    combined_segment = MidiFile(ticks_per_beat=ticks_per_beat)

    combined_track1 = MidiTrack()
    combined_track2 = MidiTrack()
    combined_segment.tracks.append(combined_track1)
    combined_segment.tracks.append(combined_track2)

    for msg in segment1.tracks[0]:
        combined_track1.append(msg.copy())

    for msg in segment2.tracks[0]:
        combined_track2.append(msg.copy())

    desired_length_ticks = int(SEGMENT_DURATION * ticks_per_beat)

    def pad_track_with_silence(track, desired_length):
        track_length = sum(msg.time for msg in track)
        if track_length < desired_length:
            silence_duration = desired_length - track_length
            track.append(Message('note_off', note=0, velocity=0, time=silence_duration))

    pad_track_with_silence(combined_track1, desired_length_ticks)
    pad_track_with_silence(combined_track2, desired_length_ticks)

    def trim_messages(track, desired_length):
        accumulated_time = 0
        for i, msg in enumerate(track):
            accumulated_time += msg.time
            if accumulated_time >= desired_length:
                msg.time -= accumulated_time - desired_length
                return track[:i+1]
        return track

    combined_track1[:] = trim_messages(combined_track1, desired_length_ticks)
    combined_track2[:] = trim_messages(combined_track2, desired_length_ticks)

    return combined_segment


def split_dataset(total_files, train_ratio):
    indices = list(range(total_files))
    random.shuffle(indices)
    split = int(train_ratio * total_files)
    return indices[:split], indices[split:]


def main():
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
        os.makedirs(TRAIN_DIR)
        os.makedirs(TEST_DIR)

    metadata = {'train': {}, 'test': {}}
    fs = FluidSynth(SOUNDFONT_PATH)

    total_files = 100
    train_indices, test_indices = split_dataset(total_files, TRAIN_SPLIT)

    for i in range(total_files):
        segment1, active_notes1 = generate_segment()
        segment2, active_notes2 = generate_segment()

        combined_segment = combine_segments(segment1, segment2)
        combined_segment_name = f"combined_segment_{i}.mid"
        combined_segment_wav = f"combined_segment_{i}.wav"
        temp_midi_path = os.path.join(ROOT_DIR, combined_segment_name)

        combined_segment.save(temp_midi_path)
        fs.midi_to_audio(temp_midi_path, os.path.join(
            ROOT_DIR, combined_segment_wav))
        os.remove(temp_midi_path)

        audio, sample_rate = sf.read(
            os.path.join(ROOT_DIR, combined_segment_wav))
        if len(audio) > int(SEGMENT_DURATION * SAMPLE_RATE):
            audio = audio[:int(SEGMENT_DURATION * SAMPLE_RATE)]
        sf.write(os.path.join(ROOT_DIR, combined_segment_wav),
                 audio, sample_rate)

        combined_notes_metadata = sorted(
            active_notes1 + active_notes2, key=lambda x: x['start_time'])

        if i in train_indices:
            shutil.move(os.path.join(
                ROOT_DIR, combined_segment_wav), TRAIN_DIR)
            metadata['train'][combined_segment_wav] = combined_notes_metadata
        else:
            shutil.move(os.path.join(ROOT_DIR, combined_segment_wav), TEST_DIR)
            metadata['test'][combined_segment_wav] = combined_notes_metadata

    for split in ['train', 'test']:
        metadata_path = os.path.join(ROOT_DIR, f"{split}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata[split], f, indent=4)


if __name__ == "__main__":
    main()
