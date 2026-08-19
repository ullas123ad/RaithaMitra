"""
RaithaMitra — Phase 5.3: Kannada Text-to-Speech (TTS) Synthesizer
================================================================
Generates natural, spoken Kannada audio from advisory text responses.
Uses neural Kannada speech synthesis with 24kHz mono PCM WAV output.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import soundfile as sf
import edge_tts

from model.tts.config import TTSConfig


class TTSGenerationError(Exception):
    """Custom exception raised when Kannada speech synthesis fails."""
    pass


class KannadaTTSEngine:
    """Neural Text-to-Speech synthesizer for Kannada agricultural advisory responses."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()

    def _run_async(self, coro):
        """Helper to run async coroutines safely across different event loop contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In an already running event loop (e.g. jupyter or async framework)
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize Kannada text into a standard 24kHz mono PCM WAV audio file.

        Args:
            text: Kannada text to speak.
            output_path: Optional path to save the generated WAV file.
            voice: Optional voice name override (e.g. 'kn-IN-GaganNeural').
            rate: Optional speaking rate override (e.g. '-10%').

        Returns:
            Dictionary containing audio metadata: audio_path, duration_seconds,
            sample_rate, format, voice, rate, latency_seconds, size_bytes.

        Raises:
            ValueError: If text is empty or invalid.
            TTSGenerationError: If synthesis or audio conversion fails.
        """
        if text is None or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        selected_voice = voice or self.config.voice
        selected_rate = rate or self.config.rate
        target_sr = self.config.sample_rate

        # Determine target output path
        if output_path is None:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            unique_id = uuid.uuid4().hex[:8]
            target_wav = out_dir / f"kannada_advisory_{unique_id}.wav"
        else:
            target_wav = Path(output_path)
            target_wav.parent.mkdir(parents=True, exist_ok=True)

        temp_mp3 = target_wav.with_suffix(f".tmp_{uuid.uuid4().hex[:6]}.mp3")

        t_start = time.time()
        try:
            # Step 1: Synthesize to temporary audio stream via edge-tts
            async def _synthesize_edge():
                comm = edge_tts.Communicate(
                    text=clean_text,
                    voice=selected_voice,
                    rate=selected_rate,
                    pitch=self.config.pitch,
                    volume=self.config.volume
                )
                await comm.save(str(temp_mp3))

            self._run_async(_synthesize_edge())

            if not temp_mp3.exists() or temp_mp3.stat().st_size == 0:
                raise TTSGenerationError(f"TTS synthesis generated an empty audio file for voice '{selected_voice}'.")

            # Step 2: Load and convert to standard 24kHz mono PCM-16 WAV
            y, sr = librosa.load(str(temp_mp3), sr=target_sr, mono=True)
            sf.write(str(target_wav), y, target_sr, subtype="PCM_16")

            latency = time.time() - t_start
            duration = len(y) / target_sr
            file_size = target_wav.stat().st_size

            return {
                "audio_path": str(target_wav.resolve()),
                "duration_seconds": round(duration, 2),
                "sample_rate": target_sr,
                "format": self.config.output_format,
                "voice": selected_voice,
                "rate": selected_rate,
                "latency_seconds": round(latency, 4),
                "size_bytes": file_size,
            }

        except ValueError:
            raise
        except Exception as e:
            raise TTSGenerationError(f"Kannada speech synthesis failed: {e}") from e
        finally:
            if temp_mp3.exists():
                try:
                    temp_mp3.unlink()
                except Exception:
                    pass


class MockKannadaSynthesizer:
    """Mock Kannada TTS synthesizer for rapid deterministic testing."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mock synthesis generating a small valid WAV file."""
        if text is None or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        selected_voice = voice or self.config.voice
        selected_rate = rate or self.config.rate
        target_sr = self.config.sample_rate

        if output_path is None:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            unique_id = uuid.uuid4().hex[:8]
            target_wav = out_dir / f"mock_kannada_advisory_{unique_id}.wav"
        else:
            target_wav = Path(output_path)
            target_wav.parent.mkdir(parents=True, exist_ok=True)

        # Generate 0.5s of silence
        import numpy as np
        samples = int(target_sr * 0.5)
        audio = np.zeros(samples, dtype=np.float32)
        sf.write(str(target_wav), audio, target_sr, subtype="PCM_16")

        return {
            "audio_path": str(target_wav.resolve()),
            "duration_seconds": 0.5,
            "sample_rate": target_sr,
            "format": "wav",
            "voice": selected_voice,
            "rate": selected_rate,
            "latency_seconds": 0.01,
            "size_bytes": target_wav.stat().st_size,
        }


# Module-level singleton
_DEFAULT_ENGINE: Optional[KannadaTTSEngine] = None


def get_tts_engine(config: Optional[TTSConfig] = None) -> KannadaTTSEngine:
    """Get or create singleton Kannada TTS engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or config is not None:
        _DEFAULT_ENGINE = KannadaTTSEngine(config=config)
    return _DEFAULT_ENGINE


def synthesize_kannada(
    text: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to synthesize Kannada text into audio.

    Args:
        text: Kannada text to speak.
        output_path: Optional output file path.
        **kwargs: Additional overrides for voice, rate, config.

    Returns:
        Audio metadata dictionary.
    """
    engine = get_tts_engine()
    return engine.synthesize(text=text, output_path=output_path, **kwargs)
