import os
import json
from midi2audio import FluidSynth as MidiFluidSynth
from mido import MidiFile, MidiTrack, Message
#TODO add styles
instruments = ["piano", "guitar", "violin", "drums", "flute", "trumpet", "saxophone", "clarinet", "synthesizer", "electric_guitar", "bass_guitar"]
notes = ["C", "D", "E", "F", "G", "A", "B"]
#TODO - Find the reason; why only couple of instruments get created
instrument_programs = {
    "Yamaha Grand Piano": "0",
    "Bright Yamaha Grand": "1",
    "Electric Piano": "2",
    "Honky Tonk": "3",
    "Rhodes EP": "4",
    "Legend EP 2": "5",
    "Harpsichord": "6",
    "Clavinet": "7",
    "Celesta": "8",
    "Glockenspiel": "9",
    "Music Box": "10",
    "Vibraphone": "11",
    "Marimba": "12",
    "Xylophone": "13",
    "Tubular Bells": "14",
    "Dulcimer": "15",
    "Drawbar Organ": "16",
    "Percussive Organ": "17",
    "Rock Organ": "18",
    "Church Organ": "19",
    "Reed Organ": "20",
    "Accordion": "21",
    "Harmonica": "22",
    "Bandoneon": "23",
    "Nylon String Guitar": "24",
    "Steel String Guitar": "25",
    "Jazz Guitar": "26",
    "Clean Guitar": "27",
    "Palm Muted Guitar": "28",
    "Overdrive Guitar": "29",
    "Distortion Guitar": "30",
    "Guitar Harmonics": "31",
    "Acoustic Bass": "32",
    "Fingered Bass": "33",
    "Picked Bass": "34",
    "Fretless Bass": "35",
    "Slap Bass": "36",
    "Pop Bass": "37",
    "Synth Bass 1": "38",
    "Synth Bass 2": "39",
    "Violin": "40",
    "Viola": "41",
    "Cello": "42",
    "Contrabass": "43",
    "Tremolo": "44",
    "Pizzicato Section": "45",
    "Harp": "46",
    "Timpani": "47",
    "Strings": "48",
    "Slow Strings": "49",
    "Synth Strings 1": "50",
    "Synth Strings 2": "51",
    "Ahh Choir": "52",
    "Ohh Voices": "53",
    "Synth Voice": "54",
    "Orchestra Hit": "55",
    "Trumpet": "56",
    "Trombone": "57",
    "Tuba": "58",
    "Muted Trumpet": "59",
    "French Horns": "60",
    "Brass Section": "61",
    "Synth Brass 1": "62",
    "Synth Brass 2": "63",
    "Soprano Sax": "64",
    "Alto Sax": "65",
    "Tenor Sax": "66",
    "Baritone Sax": "67",
    "Oboe": "68",
    "English Horn": "69",
    "Bassoon": "70",
    "Clarinet": "71",
    "Piccolo": "72",
    "Flute": "73",
    "Recorder": "74",
    "Pan Flute": "75",
    "Bottle Chiff": "76",
    "Shakuhachi": "77",
    "Whistle": "78",
    "Ocarina": "79",
    "Square Lead": "80",
    "Saw Wave": "81",
    "Calliope Lead": "83",
    "Chiffer Lead": "84",
    "Charang": "85",
    "Solo Vox": "86",
    "Fifth Sawtooth Wave": "87",
    "Bass & Lead": "88",
    "Fantasia": "89",
    "Warm Pad": "90",
    "Polysynth": "91",
    "Space Voice": "92",
    "Bowed Glass": "93",
    "Metal Pad": "94",
    "Halo Pad": "95",
    "Sweep Pad": "96",
    "Ice Rain": "97",
    "Soundtrack": "98",
    "Crystal": "99",
    "Atmosphere": "100",
    "Brightness": "101",
    "Goblin": "102",
    "Echo Drops": "103",
    "Star Theme": "104",
    "Sitar": "105",
    "Banjo": "106",
    "Shamisen": "107",
    "Koto": "108",
    "Kalimba": "109",
    "BagPipe": "110",
    "Fiddle": "111",
    "Shenai": "112",
    "Tinker Bell": "113",
    "Agogo": "114",
    "Steel Drums": "115",
    "Woodblock": "116",
    "Taiko Drum": "117",
    "Melodic Tom": "118",
    "Synth Drum": "119",
    "Reverse Cymbal": "120",
    "Fret Noise": "121",
    "Breath Noise": "122",
    "Sea Shore": "123",
    "Bird Tweet": "124",
    "Telephone": "125",
    "Helicopter": "126",
    "Applause": "127",
    "Gun Shot": "128",
    "Gtr. Cut Noise": "120",
    # ... (other noises)
    "SFX": "56"
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
