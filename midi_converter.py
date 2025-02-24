import numpy as np
import pretty_midi
import librosa


def convert_pitch_to_midi(time: np.ndarray, frequency: np.ndarray, confidence: np.ndarray,
                          output_midi_path: str) -> None:
    """Converts pitch data from CREPE into a MIDI file with better note segmentation.

    This function processes the detected pitch, filters unreliable values,
    intelligently merges consecutive notes that are meant to be sustained,
    and generates a smoother MIDI output.

    Args:
        time (np.ndarray): Array of time steps corresponding to pitch detection.
        frequency (np.ndarray): Array of detected fundamental frequencies in Hz.
        confidence (np.ndarray): Array of confidence scores for each detected pitch.
        output_midi_path (str): Path to save the generated MIDI file.

    Returns:
        None
    """
    # Convert Hz to MIDI note numbers
    midi_notes = librosa.hz_to_midi(frequency)

    # Create a new MIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=40)  # Violin (MIDI Program 40)

    # Parameters for note merging
    pitch_tolerance = 0.75  # Allow small pitch variations (in semitones) for held notes
    max_gap = 0.08  # Allow a small gap (80ms) between notes without cutting them
    min_note_duration = 0.1  # Minimum duration (100ms) to consider it a valid note

    current_note = None

    for i in range(len(time) - 1):
        if np.isnan(midi_notes[i]) or confidence[i] < 0.4:  # More strict threshold
            continue  # Skip unreliable notes

        note_number = int(round(midi_notes[i]))  # Convert to nearest MIDI note
        start_time = time[i]
        end_time = time[i + 1]

        # Ensure valid MIDI range (0-127)
        if not (0 <= note_number <= 127):
            continue

        if current_note is None:
            # Start a new note
            current_note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
        else:
            # Check if we should merge this note with the previous one
            frequency_difference = abs(librosa.midi_to_hz(current_note.pitch) - librosa.midi_to_hz(note_number))
            time_gap = start_time - current_note.end

            if frequency_difference <= librosa.midi_to_hz(pitch_tolerance) and time_gap <= max_gap:
                current_note.end = end_time  # Extend current note
            else:
                # Save the previous note if it's long enough
                if (current_note.end - current_note.start) >= min_note_duration:
                    instrument.notes.append(current_note)

                # Start a new note
                current_note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)

    # Append the last note if valid
    if current_note is not None and (current_note.end - current_note.start) >= min_note_duration:
        instrument.notes.append(current_note)

    # Add instrument to the MIDI object
    midi.instruments.append(instrument)

    # Save the MIDI file
    midi.write(output_midi_path)
