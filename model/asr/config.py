"""
ASR Module Configuration for RaithaMitra Kannada Speech Recognition.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRConfig:
    """Configuration settings for the Kannada ASR component."""

    # Model identifiers
    # Primary: vasista22/whisper-kannada-small (OpenAI Whisper fine-tuned for Kannada ASR)
    # Fallback: vasista22/whisper-kannada-tiny
    model_id: str = os.getenv("KANNADA_ASR_MODEL", "vasista22/whisper-kannada-small")
    fallback_model_id: str = "vasista22/whisper-kannada-tiny"

    # Audio parameters
    target_sampling_rate: int = 16000
    expected_language: str = "kn"
    max_audio_duration_seconds: float = 300.0  # 5 minutes max audio query

    # Model cache directory (stored in workspace saved_models or user home cache)
    cache_dir: Optional[str] = os.getenv("ASR_CACHE_DIR", None)

    # Generation parameters
    num_beams: int = 1
    use_cache: bool = True
    max_new_tokens: int = 128

    # Device configuration
    device: Optional[str] = None  # Auto-detected if None
