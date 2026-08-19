"""
Unit tests for Voice-to-Advisory Bridge (model/advisory/voice_bridge.py).
Tests end-to-end voice processing flow, error handling on non-existent files,
empty transcripts, and metadata serialization.
"""

import os
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

        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            scheme_service=self.scheme_service,
            soil_service=self.soil_service,
            market_service=self.market_service
        )
        self.dummy_audio_path = "dataset/samples/sample_kannada_query.wav"

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


if __name__ == "__main__":
    unittest.main()
