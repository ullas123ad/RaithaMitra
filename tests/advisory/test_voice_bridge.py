"""
Unit tests for Voice-to-Advisory Bridge (model/advisory/voice_bridge.py).
Tests end-to-end voice processing flow, error handling on non-existent files,
empty transcripts, TTS integration, and metadata serialization.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from model.advisory.voice_bridge import process_voice_advisory, VoiceAdvisoryError
from model.location.service import LocationService
from model.weather.service import WeatherService
from model.weather.client import MockWeatherClient
from model.soil.service import SoilService
from model.schemes.service import SchemeService
from model.market.service import MarketService
from model.market.client import MockMarketClient
from model.tts.synthesizer import MockKannadaSynthesizer
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge


class TestVoiceBridge(unittest.TestCase):
    """Test suite for process_voice_advisory function."""

    def setUp(self) -> None:
        self.loc_service = LocationService()
        self.weather_service = WeatherService(client=MockWeatherClient())
        self.soil_service = SoilService()
        self.scheme_service = SchemeService()
        self.market_service = MarketService(client=MockMarketClient())
        self.language_bridge = MockLanguageBridge()
        self.mock_tts = MockKannadaSynthesizer()

        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            scheme_service=self.scheme_service,
            soil_service=self.soil_service,
            market_service=self.market_service
        )
        self.dummy_audio_path = "dataset/samples/sample_kannada_query.wav"
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_voice_bridge_missing_file_raises_error(self) -> None:
        """Verify missing audio file raises VoiceAdvisoryError."""
        with self.assertRaises(VoiceAdvisoryError):
            process_voice_advisory(
                audio_path="non_existent_path.wav",
                advisory_engine=self.engine
            )

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_voice_bridge_empty_transcript_raises_error(self, mock_transcribe) -> None:
        """Verify empty transcript from ASR raises VoiceAdvisoryError."""
        mock_transcribe.return_value = {
            "text": "",
            "duration_seconds": 1.0,
            "processing_time_seconds": 0.1,
            "model": "vasista22/whisper-kannada-small"
        }
        with self.assertRaises(VoiceAdvisoryError):
            process_voice_advisory(
                audio_path=self.dummy_audio_path,
                advisory_engine=self.engine
            )

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_voice_bridge_successful_flow(self, mock_transcribe) -> None:
        """Verify full voice bridge flow with location and crop resolution."""
        mock_transcribe.return_value = {
            "text": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            "duration_seconds": 3.0,
            "processing_time_seconds": 0.5,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            location_service=self.loc_service,
            weather_service=self.weather_service,
            district="Mandya",
            taluk="Pandavapura",
            village="Melukote",
            crop="ragi"
        )

        self.assertIn("response", result)
        self.assertEqual(result["canonical_crop"], "ragi")
        self.assertIn("asr", result)
        self.assertEqual(result["asr"]["transcript"], "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?")
        self.assertIn("voice_pipeline_total_time_seconds", result)
        self.assertIsNotNone(result["location"])
        self.assertEqual(result["location"]["district"], "Mandya")
        self.assertEqual(result["audio"]["available"], False)

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_voice_bridge_with_tts_synthesis(self, mock_transcribe) -> None:
        """Verify voice bridge generates spoken audio when TTS is enabled."""
        mock_transcribe.return_value = {
            "text": "ರಾಗಿ ಬೆಳೆಗೆ ನೀರಿನ ನಿರ್ವಹಣೆ ಹೇಗೆ?",
            "duration_seconds": 2.5,
            "processing_time_seconds": 0.3,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }
        out_wav = os.path.join(self.temp_dir.name, "response_speech.wav")

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            tts_engine=self.mock_tts,
            synthesize_audio=True,
            output_audio_path=out_wav
        )

        self.assertIn("audio", result)
        self.assertTrue(result["audio"]["available"])
        self.assertEqual(result["audio"]["format"], "wav")
        self.assertTrue(os.path.exists(out_wav))

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_voice_bridge_tts_failure_graceful_recovery(self, mock_transcribe) -> None:
        """Verify TTS failure does not crash the advisory or transcript response."""
        mock_transcribe.return_value = {
            "text": "ರಾಗಿ ಬೆಳೆಗೆ ನೀರಿನ ನಿರ್ವಹಣೆ ಹೇಗೆ?",
            "duration_seconds": 2.5,
            "processing_time_seconds": 0.3,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }
        failing_tts = MagicMock()
        failing_tts.synthesize.side_effect = RuntimeError("Synthesis engine error")

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            tts_engine=failing_tts,
            synthesize_audio=True
        )

        self.assertIn("response", result)
        self.assertIn("asr", result)
        self.assertIn("audio", result)
        self.assertFalse(result["audio"]["available"])
        self.assertIn("Audio synthesis unavailable", result["audio"]["error"])


if __name__ == "__main__":
    unittest.main()
