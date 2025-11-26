# app/fourier/routes.py

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.fourier.basic_pitch_processor import BasicPitchProcessor

# Create a router for fourier-related endpoints
router = APIRouter(prefix="/fourier", tags=["fourier"])

# Directory to store uploaded and processed files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "langhashi_fourier"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


class AudioToMidiResponse(BaseModel):
    """Response model for audio-to-MIDI conversion."""

    filename: str
    midi_url: str
    note_count: int
    duration_seconds: float


def clean_old_files(background_tasks: BackgroundTasks, file_path: Path) -> None:
    """Schedule cleanup of temporary files.

    Args:
        background_tasks: FastAPI background tasks object
        file_path: Path to file that should be removed after response is sent
    """

    def cleanup():
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

    background_tasks.add_task(cleanup)


@router.post("/audio-to-midi", response_model=AudioToMidiResponse)
async def convert_audio_to_midi(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    min_note_duration: float = Form(0.05),
    min_frequency: float = Form(27.5),
    max_frequency: float = Form(4186.0),
):
    """Convert an audio file to MIDI using Spotify's Basic Pitch.

    Args:
        background_tasks: FastAPI background tasks
        audio_file: The audio file to convert (WAV format)
        min_note_duration: Minimum note duration in seconds
        min_frequency: Minimum frequency in Hz
        max_frequency: Maximum frequency in Hz

    Returns:
        Information about the generated MIDI file
    """
    # Validate file type (only support WAV for now)
    if not audio_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Create a unique filename
    temp_file = UPLOAD_DIR / f"upload_{audio_file.filename}"

    try:
        # Save the uploaded file
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # Process the audio file
        processor = BasicPitchProcessor()
        result = processor.process_audio_file(
            str(temp_file),
            output_dir=str(UPLOAD_DIR),
            save_midi=True,
            min_note_duration=min_note_duration,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

        if "midi_path" not in result:
            raise HTTPException(status_code=500, detail="Failed to generate MIDI file")

        midi_path = result["midi_path"]
        midi_filename = os.path.basename(midi_path)

        # Get note count and duration from the MIDI data
        midi_data = result["midi_data"]
        note_count = sum(len(instrument.notes) for instrument in midi_data.instruments)
        duration_seconds = midi_data.get_end_time()

        # Schedule cleanup
        clean_old_files(background_tasks, temp_file)

        return AudioToMidiResponse(
            filename=midi_filename,
            midi_url=f"/fourier/download/{midi_filename}",
            note_count=note_count,
            duration_seconds=duration_seconds,
        )

    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            os.unlink(temp_file)
        raise HTTPException(
            status_code=500, detail=f"Error processing audio: {str(e)}"
        ) from e


@router.get("/download/{filename}")
async def download_midi_file(background_tasks: BackgroundTasks, filename: str):
    """Download a generated MIDI file.

    Args:
        background_tasks: FastAPI background tasks
        filename: The name of the MIDI file to download

    Returns:
        The MIDI file as a download
    """
    file_path = UPLOAD_DIR / filename

    if not file_path.exists() or not filename.lower().endswith(".mid"):
        raise HTTPException(status_code=404, detail="MIDI file not found")

    # Schedule file for cleanup after a reasonable time
    # We don't delete immediately so user has time to download
    def delayed_cleanup():
        import time

        time.sleep(300)  # Keep file for 5 minutes
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

    background_tasks.add_task(delayed_cleanup)

    return FileResponse(path=file_path, media_type="audio/midi", filename=filename)


@router.get("/health")
async def health_check():
    """Health check endpoint for the Fourier module."""
    return {"status": "healthy", "service": "fourier-audio-processing"}
