"""
RaithaMitra — Phase 5.3: Kannada TTS Module
===========================================
Text-to-Speech synthesis for Kannada spoken agricultural advisories.
"""

from model.tts.config import TTSConfig
from model.tts.synthesizer import (
    KannadaTTSEngine,
    MockKannadaSynthesizer,
    TTSGenerationError,
    get_tts_engine,
    synthesize_kannada,
)

__all__ = [
    "TTSConfig",
    "KannadaTTSEngine",
    "MockKannadaSynthesizer",
    "TTSGenerationError",
    "get_tts_engine",
    "synthesize_kannada",
]
