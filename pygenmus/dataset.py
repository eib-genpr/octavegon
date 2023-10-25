import os
import json
import soundfile as sf
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message
import random

# Constants
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
SEGMENT_DURATION = 7.0
SAMPLE_RATE = 44100


def random_duration(total_duration, segment_duration):
    max_duration = segment_duration - total_duration
    return round(random.uniform(0.1, min(2.0, max_duration)), 3)


def generate_segment():
    midi_file = MidiFile()
    midi_track = MidiTrack()
    midi_file.tracks.append(midi_track)

    total_duration = 0.0
    program = 0
    active_notes = {}  # Track active notes by their note value

    while total_duration < SEGMENT_DURATION:
        instrument = random.choice(INSTRUMENTS)
        note = random.choice(NOTES)
        duration = random_duration(
            total_duration=total_duration, segment_duration=SEGMENT_DURATION)

        programs = INSTRUMENT_PROGRAMS.get(instrument, [])
        program = random.choice(programs) if programs else 0

        note_value = NOTES.index(note) + 60
        velocity = random.randint(80, 120)
        duration_ms = int(duration * 1000)

        start_timestamp = total_duration
        end_timestamp = total_duration + duration

        midi_track.append(Message("program_change", program=program))
        midi_track.append(Message("note_on", note=note_value,
                          velocity=velocity, time=duration_ms))

        if end_timestamp > SEGMENT_DURATION:
            # Adjust duration
            duration_ms = int((SEGMENT_DURATION - start_timestamp) * 1000)
            end_timestamp = SEGMENT_DURATION

        midi_track.append(
            Message("note_off", note=note_value, time=duration_ms))

        # Handle active notes using a dictionary
        if instrument in active_notes:
            active_notes[instrument].append({
                "note": note,
                "start_timestamp": round(start_timestamp, 3),
                "end_timestamp": round(end_timestamp, 3),
                "duration": round(end_timestamp - start_timestamp, 3)
            })
        else:
            active_notes[instrument] = [{
                "note": note,
                "start_timestamp": round(start_timestamp, 3),
                "end_timestamp": round(end_timestamp, 3),
                "duration": round(end_timestamp - start_timestamp, 3)
            }]

        total_duration = end_timestamp

    return midi_file, active_notes


def combine_segments(segment1, segment2):
    combined_segment = MidiFile()
    combined_segment.tracks.append(segment1.tracks[0])
    combined_segment.tracks.append(segment2.tracks[0])
    return combined_segment


def main():
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    metadata = {}
    fs = MidiFluidSynth("C:\\soundfonts\\FluidR3_GM.sf2")

    for i in range(100):
        segment1, active_notes1 = generate_segment()
        segment2, active_notes2 = generate_segment()

        combined_segment = combine_segments(segment1, segment2)

        combined_segment_name = f"combined_segment_{i}.wav"
        combined_file_path = os.path.join(ROOT_DIR, combined_segment_name)

        temp_midi_path = os.path.join(
            ROOT_DIR, f"temp_{combined_segment_name}.mid")

        combined_segment.save(temp_midi_path)
        fs.midi_to_audio(temp_midi_path, combined_file_path)
        os.remove(temp_midi_path)

        audio, sample_rate = sf.read(combined_file_path)
        if len(audio) > int(SEGMENT_DURATION * sample_rate):
            audio = audio[:int(SEGMENT_DURATION * sample_rate)]

        sf.write(combined_file_path, audio, sample_rate)

        metadata[combined_segment_name] = {**active_notes1, **active_notes2}

    metadata_path = os.path.join(ROOT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
