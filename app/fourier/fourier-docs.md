# Fourier API Documentation

This document provides a brief overview of the Fourier API endpoints for audio processing.

## Endpoints

### `POST /fourier/audio-to-midi`

Converts an audio file to a MIDI file using Spotify's Basic Pitch algorithm.

**Request Body:**

The request should be a `multipart/form-data` request with the following fields:

- `audio_file`: The audio file to convert. **Only WAV files are currently supported.**
- `min_note_duration` (optional, float, default: 0.05): The minimum duration of a note in seconds.
- `min_frequency` (optional, float, default: 27.5): The minimum frequency of a note in Hz.
- `max_frequency` (optional, float, default: 4186.0): The maximum frequency of a note in Hz.

**Responses:**

- **200 OK:**
  - **Content-Type:** `application/json`
  - **Body:**
    ```json
    {
      "filename": "your_audio.mid",
      "midi_url": "/fourier/download/your_audio.mid",
      "note_count": 123,
      "duration_seconds": 45.67
    }
    ```

- **400 Bad Request:**
  - If the uploaded file is not a WAV file.
  - **Body:** `{"detail": "Only WAV files are supported"}`

- **500 Internal Server Error:**
  - If there is an error during the audio processing.
  - **Body:** `{"detail": "Error processing audio: <error_message>"}`

---

### `GET /fourier/download/{filename}`

Downloads a generated MIDI file.

**Path Parameters:**

- `filename` (string, required): The name of the MIDI file to download, as returned by the `/audio-to-midi` endpoint.

**Responses:**

- **200 OK:**
  - **Content-Type:** `audio/midi`
  - The MIDI file is returned as an attachment.

- **404 Not Found:**
  - If the requested MIDI file does not exist.
  - **Body:** `{"detail": "MIDI file not found"}`

---

### `GET /fourier/health`

A health check endpoint to verify that the Fourier module is running.

**Responses:**

- **200 OK:**
  - **Content-Type:** `application/json`
  - **Body:**
    ```json
    {
      "status": "healthy",
      "service": "fourier-audio-processing"
    }
    ```
