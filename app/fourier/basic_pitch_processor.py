# app/fourier/basic_pitch_processor.py

import os
import tempfile

import numpy as np
import pretty_midi
from basic_pitch.inference import predict


class BasicPitchProcessor:
    """Process audio files using Spotify's Basic Pitch library.

    This class handles the conversion of .wav audio files to MIDI
    using Spotify's Basic Pitch algorithm, which detects notes and
    pitch bends in musical audio.
    """

    def __init__(self, model_path: str | None = None):
        """Initialize the BasicPitchProcessor.

        Args:
            model_path: Path to the Basic Pitch model. If None, uses the default model.
        """
        self.model_path = model_path

    def process_audio_file(
        self,
        audio_path: str,
        output_dir: str | None = None,
        save_midi: bool = True,
        sonify_midi: bool = False,
        min_note_duration: float = 0.05,
        min_frequency: float = 27.5,
        max_frequency: float = 4186.0,
    ) -> dict:
        """Process a WAV audio file and convert it to MIDI.

        Args:
            audio_path: Path to the input WAV file.
            output_dir: Directory to save output files. If None, uses a temporary directory.
            save_midi: Whether to save the MIDI file to disk.
            sonify_midi: Whether to save a sonification of the MIDI file.
            min_note_duration: Minimum note duration in seconds.
            min_frequency: Minimum frequency in Hz.
            max_frequency: Maximum frequency in Hz.

        Returns:
            Dict containing processed data and file paths.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Get filename without extension
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]

        # Define output paths
        midi_path = os.path.join(output_dir, f"{audio_basename}.mid")

        # Run Basic Pitch prediction
        model_output, midi_data, note_events = self._run_prediction(
            audio_path,
            midi_path if save_midi else None,
            sonify_midi,
            min_note_duration,
            min_frequency,
            max_frequency,
        )

        result = {
            "model_output": model_output,
            "midi_data": midi_data,
            "note_events": note_events,
        }

        if save_midi:
            result["midi_path"] = midi_path

        return result

    def _run_prediction(
        self,
        audio_path: str,
        midi_path: str | None = None,
        sonify_midi: bool = False,
        min_note_duration: float = 0.05,
        min_frequency: float = 27.5,
        max_frequency: float = 4186.0,
    ) -> tuple[dict, pretty_midi.PrettyMIDI, list]:
        """Run the Basic Pitch prediction algorithm.

        Args:
            audio_path: Path to the input WAV file.
            midi_path: Path to save the MIDI file. If None, MIDI is not saved.
            sonify_midi: Whether to save a sonification of the MIDI file.
            min_note_duration: Minimum note duration in seconds.
            min_frequency: Minimum frequency in Hz.
            max_frequency: Maximum frequency in Hz.

        Returns:
            Tuple containing model outputs, MIDI data, and note events.
        """
        # Run Basic Pitch prediction - new API returns tuple directly
        # Parameters: minimum_note_length is in milliseconds
        predict_args = {
            "audio_path": audio_path,
            "minimum_note_length": min_note_duration * 1000,  # Convert to milliseconds
            "minimum_frequency": min_frequency,
            "maximum_frequency": max_frequency,
        }
        if self.model_path:
            predict_args["model_or_model_path"] = self.model_path

        model_output, midi_data, note_events = predict(**predict_args)

        # Save MIDI if path is provided
        if midi_path is not None:
            midi_data.write(midi_path)

            # Create sonification if requested
            if sonify_midi:
                # Basic Pitch doesn't have direct sonification, but pretty_midi does
                audio_data = midi_data.fluidsynth(fs=44100)
                # Normalize audio
                audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                # Here you would save the audio_data as a WAV file
                # We're not implementing this part since it requires additional dependencies

        return model_output, midi_data, note_events


def process_wav_to_midi(
    wav_path: str,
    output_dir: str | None = None,
    save_midi: bool = True,
    min_note_duration: float = 0.05,
) -> dict:
    """Process a WAV file and return MIDI data.

    Args:
        wav_path: Path to the input WAV file.
        output_dir: Directory to save output files. If None, uses a temporary directory.
        save_midi: Whether to save the MIDI file to disk.
        min_note_duration: Minimum note duration in seconds.

    Returns:
        Dict containing processed data and file paths.
    """
    processor = BasicPitchProcessor()
    return processor.process_audio_file(
        wav_path, output_dir=output_dir, save_midi=save_midi, min_note_duration=min_note_duration
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process WAV files to MIDI using Spotify's Basic Pitch"
    )
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument("--output-dir", "-o", help="Output directory for MIDI files")
    parser.add_argument(
        "--min-duration", "-d", type=float, default=0.05, help="Minimum note duration in seconds"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save MIDI file to disk")

    args = parser.parse_args()

    result = process_wav_to_midi(
        args.input,
        output_dir=args.output_dir,
        save_midi=not args.no_save,
        min_note_duration=args.min_duration,
    )

    if "midi_path" in result:
        print(f"MIDI file saved to: {result['midi_path']}")
    else:
        print("MIDI data processed but not saved to disk")
