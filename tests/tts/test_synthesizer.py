"""
Unit Tests for Kannada Text-to-Speech (TTS) Synthesizer
======================================================
Verifies:
1. Module imports and symbol export.
2. Configuration defaults and custom options.
3. Input validation (empty text, None, non-string).
4. Mock synthesizer generation, output path, metadata schema.
5. Error handling and failure recovery.
6. Singleton engine access.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from model.tts.config import TTSConfig
from model.tts.synthesizer import (
    KannadaTTSEngine,
    MockKannadaSynthesizer,
    TTSGenerationError,
    get_tts_engine,
    synthesize_kannada,
)


class TestKannadaTTS(unittest.TestCase):
    """Test suite for Kannada TTS module."""

    def setUp(self):
        self.config = TTSConfig(
            voice="kn-IN-GaganNeural",
            alternate_voice="kn-IN-SapnaNeural",
            rate="-10%",
            sample_rate=24000,
            output_format="wav"
        )
        self.mock_synthesizer = MockKannadaSynthesizer(config=self.config)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tts_module_exports(self):
        """Verify module exports all required classes and helpers."""
        self.assertTrue(issubclass(KannadaTTSEngine, object))
        self.assertTrue(issubclass(MockKannadaSynthesizer, object))
        self.assertTrue(issubclass(TTSGenerationError, Exception))
        self.assertTrue(callable(get_tts_engine))
        self.assertTrue(callable(synthesize_kannada))

    def test_tts_config_defaults(self):
        """Verify default configuration attributes."""
        default_cfg = TTSConfig()
        self.assertEqual(default_cfg.voice, "kn-IN-GaganNeural")
        self.assertEqual(default_cfg.alternate_voice, "kn-IN-SapnaNeural")
        self.assertEqual(default_cfg.rate, "-10%")
        self.assertEqual(default_cfg.sample_rate, 24000)
        self.assertEqual(default_cfg.output_format, "wav")

    def test_mock_synthesizer_success(self):
        """Verify MockKannadaSynthesizer generates valid audio metadata and file."""
        text = "ನೀರಿನ ಒತ್ತಡವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಸೂಕ್ತ ನೀರಾವರಿ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ."
        out_path = os.path.join(self.temp_dir.name, "test_advisory.wav")
        result = self.mock_synthesizer.synthesize(text=text, output_path=out_path)

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertEqual(result["format"], "wav")
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(result["voice"], "kn-IN-GaganNeural")
        self.assertGreater(result["duration_seconds"], 0)
        self.assertIn("latency_seconds", result)

    def test_mock_synthesizer_empty_text_raises_value_error(self):
        """Verify empty or whitespace string input raises ValueError."""
        with self.assertRaises(ValueError):
            self.mock_synthesizer.synthesize("")

        with self.assertRaises(ValueError):
            self.mock_synthesizer.synthesize("   \n\t  ")

    def test_mock_synthesizer_none_input_raises_value_error(self):
        """Verify None or non-string input raises ValueError."""
        with self.assertRaises(ValueError):
            self.mock_synthesizer.synthesize(None)

        with self.assertRaises(ValueError):
            self.mock_synthesizer.synthesize(12345)

    def test_mock_synthesizer_default_directory_generation(self):
        """Verify synthesis creates output file in default directory when output_path is None."""
        text = "ರಾಗಿ ಬೆಳೆಗೆ ಕಳೆ ನಿರ್ವಹಣೆ ಅಗತ್ಯ."
        result = self.mock_synthesizer.synthesize(text=text)
        created_path = result.get("audio_path")

        self.assertTrue(os.path.exists(created_path))
        # Cleanup created default test artifact
        try:
            os.remove(created_path)
        except Exception:
            pass

    def test_kannada_tts_engine_validation(self):
        """Verify KannadaTTSEngine validates text inputs prior to execution."""
        engine = KannadaTTSEngine(config=self.config)
        with self.assertRaises(ValueError):
            engine.synthesize("")

        with self.assertRaises(ValueError):
            engine.synthesize(None)

    def test_kannada_tts_engine_failure_wraps_in_tts_error(self):
        """Verify internal synthesis failures raise TTSGenerationError."""
        engine = KannadaTTSEngine(config=self.config)
        with patch("edge_tts.Communicate", side_effect=RuntimeError("Network failure")):
            with self.assertRaises(TTSGenerationError):
                engine.synthesize("ಟೆಸ್ಟ್ ವಾಕ್ಯ")

    def test_singleton_engine_getter(self):
        """Verify get_tts_engine returns a valid KannadaTTSEngine instance."""
        engine1 = get_tts_engine()
        engine2 = get_tts_engine()
        self.assertIs(engine1, engine2)
        self.assertIsInstance(engine1, KannadaTTSEngine)


if __name__ == "__main__":
    unittest.main()
