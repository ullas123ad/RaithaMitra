"""
RaithaMitra — Phase 5.3: Kannada TTS Configuration
==================================================
Configuration for Kannada Text-to-Speech synthesis.
"""

from dataclasses import dataclass


@dataclass
class TTSConfig:
    """Configuration parameters for Kannada speech synthesis."""

    # Default voice: kn-IN-GaganNeural (clear, authoritative Kannada male voice)
    voice: str = "kn-IN-GaganNeural"

    # Alternate voice: kn-IN-SapnaNeural (Kannada female voice)
    alternate_voice: str = "kn-IN-SapnaNeural"

    # Speech rate: -10% gives natural, farmer-friendly pacing
    rate: str = "-10%"

    # Pitch adjustment
    pitch: str = "+0Hz"

    # Volume adjustment
    volume: str = "+0%"

    # Output audio sample rate in Hz (standard 24kHz)
    sample_rate: int = 24000

    # Output format
    output_format: str = "wav"

    # Default output directory for generated speech
    output_dir: str = "outputs"
