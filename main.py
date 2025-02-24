import crepe
import librosa
import os
from midi_converter import convert_pitch_to_midi

# Path to the WAV file
audio_path = "/Users/dylan/Downloads/12-Chemin-Pierre-de-Ronsard-7.wav"

# Extract filename without extension and directory
base_name = os.path.splitext(os.path.basename(audio_path))[0]  # Example: "12-Chemin-Pierre-de-Ronsard-7"
directory = os.path.dirname(audio_path)  # Gets the folder path

# Define the MIDI output path in the same directory with the same name
midi_path = os.path.join(directory, f"{base_name}.mid")

# Load audio
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Pitch detection with CREPE
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=10)

# Display first 10 detected pitches
print("Detected frequencies (Hz):", frequency[:10])
print("Confidence scores:", confidence[:10])

# Convert detected pitch to MIDI
convert_pitch_to_midi(time, frequency, confidence, midi_path)

print(f"MIDI file saved: {midi_path}")
