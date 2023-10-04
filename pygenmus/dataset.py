import os
import json
import soundfile as sf
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message
import librosa
import numpy as np

# Define constants
#//TODO drums do not have notes
INSTRUMENTS = ["piano", "guitar", "violin", "drums", "flute", "trumpet",
               "saxophone", "clarinet", "synthesizer", "electric_guitar", "bass_guitar"]
NOTES = ["C", "D", "E", "F", "G", "A", "B"]

#//TODO Add more instruments and styles
INSTRUMENT_PROGRAMS = {
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

ROOT_DIR = "dataset_root"
AUGMENT = True
OVERLAP = True
SEGMENT_LENGTH = 44100 * 3
SAMPLE_RATE = 44100
VOLUME_FACTOR = 10.0


def load_audio_data(file_path, segment_length, overlap=False):
    audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    num_segments = len(audio_data) // segment_length

    if overlap:
        step_size = segment_length // 2
    else:
        step_size = segment_length

    for i in range(0, len(audio_data) - segment_length + 1, step_size):
        audio_segment = audio_data[i:i + segment_length]
        audio_segment *= VOLUME_FACTOR

        yield audio_segment.reshape(-1, 1)


def main():
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    for instrument in INSTRUMENTS:
        instrument_dir = os.path.join(ROOT_DIR, instrument)
        os.makedirs(instrument_dir, exist_ok=True)
        for note in NOTES:
            note_dir = os.path.join(instrument_dir, note)
            os.makedirs(note_dir, exist_ok=True)

    metadata = {}
    fs = MidiFluidSynth()

    # Overlapping
    for instrument1 in INSTRUMENTS:
        for note1 in NOTES:
            for instrument2 in INSTRUMENTS:
                for note2 in NOTES:
                    if instrument1 != instrument2 or note1 != note2:
                        file_name = f"{instrument1}_{note1}_{instrument2}_{note2}.wav"
                        file_path = os.path.join(
                            ROOT_DIR, instrument1, note1, file_name)

                        midi_file = MidiFile()

                        midi_track1 = MidiTrack()
                        midi_track2 = MidiTrack()
                        midi_file.tracks.extend([midi_track1, midi_track2])

                        program1 = INSTRUMENT_PROGRAMS.get(instrument1, 0)
                        midi_track1.append(
                            Message("program_change", program=program1))

                        note_value1 = NOTES.index(note1) + 60
                        velocity = 100
                        duration = 1000

                        midi_track1.append(
                            Message("note_on", note=note_value1, velocity=velocity))
                        midi_track1.append(
                            Message("note_off", note=note_value1, time=duration))

                        program2 = INSTRUMENT_PROGRAMS.get(instrument2, 0)
                        midi_track2.append(
                            Message("program_change", program=program2))

                        note_value2 = NOTES.index(note2) + 60
                        midi_track2.append(
                            Message("note_on", note=note_value2, velocity=velocity))
                        midi_track2.append(
                            Message("note_off", note=note_value2, time=duration))

                        temp_midi_path = os.path.join(
                            instrument1, note1, f"temp_{file_name}.mid")
                        temp_midi_path = os.path.join(ROOT_DIR, temp_midi_path)
                        midi_file.save(temp_midi_path)

                        fs.midi_to_audio(temp_midi_path, file_path)
                        os.remove(temp_midi_path)

                        audio_segments1 = list(load_audio_data(
                            file_path, SEGMENT_LENGTH, OVERLAP))
                        audio_segments2 = list(load_audio_data(
                            file_path, SEGMENT_LENGTH, OVERLAP))

                        audio_segments = [
                            seg1 + seg2 for seg1, seg2 in zip(audio_segments1, audio_segments2)]

                        sf.write(file_path, np.vstack(
                            audio_segments), SAMPLE_RATE)

                        metadata[file_name] = {
                            "instrument1": instrument1, "note1": note1, "instrument2": instrument2, "note2": note2}

    # Non-overlapping single instrument single note files //TODO reduce duplicate instrument and notes
    for instrument in INSTRUMENTS:
        for note in NOTES:
            file_name = f"{instrument}_{note}.wav"
            file_path = os.path.join(ROOT_DIR, instrument, note, file_name)

            midi_file = MidiFile()
            midi_track = MidiTrack()
            midi_file.tracks.append(midi_track)

            program = INSTRUMENT_PROGRAMS.get(instrument, 0)
            midi_track.append(
                Message("program_change", program=program))

            note_value = NOTES.index(note) + 60
            velocity = 100
            duration = 1000

            midi_track.append(
                Message("note_on", note=note_value, velocity=velocity))
            midi_track.append(
                Message("note_off", note=note_value, time=duration))

            temp_midi_path = os.path.join(
                instrument, note, f"temp_{file_name}.mid")
            temp_midi_path = os.path.join(ROOT_DIR, temp_midi_path)
            midi_file.save(temp_midi_path)

            fs.midi_to_audio(temp_midi_path, file_path)
            os.remove(temp_midi_path)

            metadata[file_name] = {
                "instrument1": instrument, "note1": note}

    metadata_path = os.path.join(ROOT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
