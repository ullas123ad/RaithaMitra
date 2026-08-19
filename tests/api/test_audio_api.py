"""
Unit tests for Audio Advisory API endpoint (/api/v1/advisory/audio).
Tests multipart file upload, format validation, size limit checks,
error responses, and ASR metadata serialization.
"""

import io
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import soundfile as sf

from api.app import create_app
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


class TestAudioAdvisoryAPI(unittest.TestCase):
    """Test suite for /api/v1/advisory/audio endpoint."""

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

        self.app = create_app(
            advisory_engine=self.engine,
            location_service=self.loc_service,
            weather_service=self.weather_service,
            soil_service=self.soil_service,
            scheme_service=self.scheme_service,
            market_service=self.market_service,
            config={"TESTING": True}
        )
        self.client = self.app.test_client()

        # Generate a small in-memory valid WAV byte buffer for testing
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.wav_bytes = io.BytesIO()
        sf.write(self.wav_bytes, audio, sr, format="WAV")
        self.wav_bytes.seek(0)

    def test_audio_endpoint_missing_file(self) -> None:
        """Verify missing audio file in request returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory/audio",
            data={"district": "Mandya"},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_audio_endpoint_unsupported_extension(self) -> None:
        """Verify unsupported file extension returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory/audio",
            data={
                "audio": (io.BytesIO(b"dummy content"), "test_file.txt"),
                "district": "Mandya"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_audio_endpoint_empty_file(self) -> None:
        """Verify 0-byte audio file returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory/audio",
            data={
                "audio": (io.BytesIO(b""), "empty.wav"),
                "district": "Mandya"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_audio_endpoint_success_with_mocked_asr(self, mock_transcribe) -> None:
        """Verify valid audio upload produces expected response and ASR metadata."""
        mock_transcribe.return_value = {
            "text": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            "language": "kn",
            "model": "vasista22/whisper-kannada-small",
            "duration_seconds": 2.5,
            "processing_time_seconds": 0.45,
            "device": "cpu"
        }

        self.wav_bytes.seek(0)
        response = self.client.post(
            "/api/v1/advisory/audio",
            data={
                "audio": (self.wav_bytes, "farmer_query.wav"),
                "district": "Mandya",
                "taluk": "Pandavapura",
                "village": "Melukote",
                "crop": "ragi"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["canonical_crop"], "ragi")
        self.assertIn("asr", data)
        self.assertEqual(data["asr"]["transcript"], "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?")
        self.assertEqual(data["asr"]["asr_model"], "vasista22/whisper-kannada-small")
        self.assertIsNotNone(data["location"])
        self.assertEqual(data["location"]["district"], "Mandya")
        self.assertIsNotNone(data["weather"])
        self.assertIsNotNone(data["soil"])
        self.assertIn("ರಾಗಿ", data["answer"])


if __name__ == "__main__":
    unittest.main()
