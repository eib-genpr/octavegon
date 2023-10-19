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
SEGMENT_DURATION = 7.0
SAMPLE_RATE = 44100


def random_duration(segment_duration, total_duration):
    max_duration = segment_duration - total_duration
    return round(random.uniform(0.1, min(2.0, max_duration)), 3)


def handle_overlapping_notes(active_notes, end_timestamp):
    notes_to_remove = [note for note, start_timestamp in active_notes.items()
                       if start_timestamp < end_timestamp]
    for note in notes_to_remove:
        del active_notes[note]


def generate_segment():
    midi_file = MidiFile()
    midi_track = MidiTrack()
    midi_file.tracks.append(midi_track)

    total_duration = 0.0
    active_notes = {}
    program = 0  # Initialize program

    while total_duration < SEGMENT_DURATION:
        instrument = random.choice(INSTRUMENTS)
        note = random.choice(NOTES)
        duration = random_duration(
            total_duration=total_duration, segment_duration=SEGMENT_DURATION)

        program = INSTRUMENT_PROGRAMS.get(instrument, 0)
        note_value = NOTES.index(note) + 60
        velocity = 100
        duration_ms = int(duration * 1000)

        start_timestamp = total_duration
        end_timestamp = total_duration + duration
        handle_overlapping_notes(active_notes, start_timestamp)

        midi_track.append(Message("program_change", program=program))
        midi_track.append(
            Message("note_on", note=note_value, velocity=velocity))

        if end_timestamp > SEGMENT_DURATION:
            # Adjust duration
            duration_ms = int((SEGMENT_DURATION - start_timestamp) * 1000)
            end_timestamp = SEGMENT_DURATION

        midi_track.append(
            Message("note_off", note=note_value, time=duration_ms))
        active_notes[note_value] = start_timestamp
        total_duration = end_timestamp

    return midi_file


def combine_segments(segment1, segment2):
    combined_segment = MidiFile()
    combined_segment.tracks.append(segment1.tracks[0])
    combined_segment.tracks.append(segment2.tracks[0])
    return combined_segment


def calculate_total_time(track):
    total_time = 0
    active_notes = {}
    for msg in track:
        if msg.type == "note_on":
            note = msg.note
            if note in active_notes:
                start_timestamp = active_notes[note]
                duration = (msg.time / 1000.0)
                active_notes[note] = start_timestamp
            else:
                active_notes[note] = 0.0
        elif msg.type == "note_off":
            note = msg.note
            if note in active_notes:
                start_timestamp = active_notes[note]
                duration = (msg.time / 1000.0)
                total_time += duration
                del active_notes[note]
    return total_time


def trim_segment(segment):
    total_time = calculate_total_time(segment.tracks[0])
    if total_time > SEGMENT_DURATION:
        for track in segment.tracks:
            current_time = 0.0
            events_to_keep = []
            for msg in track:
                if current_time >= SEGMENT_DURATION:
                    break

                if msg.type == "note_on":
                    note = msg.note
                    start_time = current_time
                    duration = msg.time / 1000.0
                    end_time = start_time + duration

                    if end_time > SEGMENT_DURATION:
                        duration = SEGMENT_DURATION - start_time
                        msg.time = int(duration * 1000)
                        events_to_keep.append(msg)
                        break

                    current_time = end_time
                    events_to_keep.append(msg)

                elif msg.type == "note_off":
                    if events_to_keep and events_to_keep[-1].note == msg.note:
                        events_to_keep[-1].time += msg.time

            track.clear()
            track.extend(events_to_keep)


def main():
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    metadata = {}
    fs = MidiFluidSynth("C:\\soundfonts\\FluidR3_GM.sf2")

    for i in range(100):
        segment1, segment2 = None, None

        while segment1 is None or calculate_total_time(segment1.tracks[0]) > SEGMENT_DURATION:
            segment1 = generate_segment()
            if calculate_total_time(segment1.tracks[0]) > SEGMENT_DURATION:
                trim_segment(segment1)

        while segment2 is None or calculate_total_time(segment2.tracks[0]) > SEGMENT_DURATION:
            segment2 = generate_segment()
            if calculate_total_time(segment2.tracks[0]) > SEGMENT_DURATION:
                trim_segment(segment2)

        combined_segment = combine_segments(segment1, segment2)

        combined_segment_name = f"combined_segment_{i}.wav"
        combined_file_path = os.path.join(ROOT_DIR, combined_segment_name)

        temp_midi_path = os.path.join(
            ROOT_DIR, f"temp_{combined_segment_name}.mid")

        instrument_metadata = {}

        current_time = 0.0
        active_notes = {}
        midi_track = MidiTrack()

        for track in combined_segment.tracks:
            program = None
            instrument_name = None

            for msg in track:
                if hasattr(msg, 'program'):
                    program = msg.program
                    instrument_name = INSTRUMENTS[program % len(INSTRUMENTS)]

                if msg.type == "note_on":
                    note = msg.note

                    if note in active_notes:
                        start_time, _ = active_notes[note]
                        duration = current_time - start_time
                        if duration > 0:
                            instrument_metadata.setdefault(instrument_name, {}).setdefault(
                                NOTES[note % 12], []).append({
                                    "duration": min(round(duration, 3), SEGMENT_DURATION - current_time),
                                    "start_timestamp": round(start_time, 3),
                                    "end_timestamp": round(min(current_time, SEGMENT_DURATION), 3)
                                })

                    start_time = current_time
                    active_notes[note] = (start_time, None)

                elif msg.type == "note_off":
                    note = msg.note

                    start_time, previous_duration = active_notes[note]
                    if previous_duration is not None and previous_duration > 0:
                        instrument_metadata.setdefault(instrument_name, {}).setdefault(
                            NOTES[note % 12], []).append({
                                "duration": min(round(previous_duration, 3), SEGMENT_DURATION - current_time),
                                "start_timestamp": round(start_time, 3),
                                "end_timestamp": round(min(current_time, SEGMENT_DURATION), 3)
                            })
                    current_time += msg.time / 1000.0

                midi_track.append(msg)

        combined_segment.tracks.append(midi_track)
        combined_segment.save(temp_midi_path)
        fs.midi_to_audio(temp_midi_path, combined_file_path)
        os.remove(temp_midi_path)

        audio, sample_rate = sf.read(combined_file_path)
        if len(audio) > int(SEGMENT_DURATION * sample_rate):
            audio = audio[:int(SEGMENT_DURATION * sample_rate)]

        sf.write(combined_file_path, audio, sample_rate)

        metadata[combined_segment_name] = instrument_metadata

    metadata_path = os.path.join(ROOT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
