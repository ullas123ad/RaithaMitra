"""
Audio Preprocessing Module for RaithaMitra ASR.

Handles audio loading, stereo-to-mono conversion, resampling to 16kHz,
normalization, and error validation without modifying original source files.
"""

import os
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa


class AudioProcessingError(Exception):
    """Custom exception raised for invalid or corrupted audio processing."""
    pass


def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_duration_seconds: float = 300.0
) -> Tuple[np.ndarray, float]:
    """
    Loads an audio file, converts it to mono, resamples to target_sr (16kHz),
    normalizes signal values, and returns the float32 array and duration.

    Args:
        audio_path: Path to the input audio file (.wav, .mp3, .flac, .ogg).
        target_sr: Target sampling rate required by ASR model (default: 16000 Hz).
        max_duration_seconds: Maximum allowed audio length in seconds.

    Returns:
        Tuple[np.ndarray, float]:
            - speech_array: 1D float32 numpy array normalized to [-1.0, 1.0].
            - duration_seconds: Total duration of the processed audio in seconds.

    Raises:
        AudioProcessingError: If file is missing, corrupted, empty, or unreadable.
    """
    # 1. Validate file existence
    if not os.path.exists(audio_path):
        raise AudioProcessingError(f"Audio file not found at path: '{audio_path}'")

    if not os.path.isfile(audio_path):
        raise AudioProcessingError(f"Path is not a valid file: '{audio_path}'")

    # 2. Check file size (non-zero)
    if os.path.getsize(audio_path) == 0:
        raise AudioProcessingError(f"Audio file is empty (0 bytes): '{audio_path}'")

    # 3. Read audio file
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Convert multi-channel to mono
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        speech_array = data
    except Exception:
        # Fallback to librosa if soundfile cannot parse container format
        try:
            speech_array, sr = librosa.load(audio_path, sr=target_sr, mono=True, dtype=np.float32)
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to read or decode audio file '{audio_path}': {str(e)}"
            )

    # 4. Check for non-empty audio array
    if speech_array is None or len(speech_array) == 0:
        raise AudioProcessingError(f"Audio file contains no decodable audio frames: '{audio_path}'")

    # 5. Compute audio duration
    duration_seconds = float(len(speech_array)) / float(target_sr)

    if duration_seconds < 0.1:
        raise AudioProcessingError(
            f"Audio duration ({duration_seconds:.2f}s) is too short for speech recognition."
        )

    if duration_seconds > max_duration_seconds:
        raise AudioProcessingError(
            f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed limit ({max_duration_seconds}s)."
        )

    # 6. Ensure finite float32 values (handle NaN/Inf safely)
    speech_array = np.nan_to_num(speech_array, nan=0.0, posinf=1.0, neginf=-1.0)

    # 7. Dynamic range normalization (prevent clipping / extreme silence distortion)
    max_val = np.max(np.abs(speech_array))
    if max_val > 1.0:
        speech_array = speech_array / max_val

    return speech_array, duration_seconds
