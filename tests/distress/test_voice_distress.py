"""
End-to-End Voice Distress & Spoken Audio Test Suite
===================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Tests:
1. HIGH distress speech input triggers fast-path, bypassing LLM/RAG, and synthesizes spoken Kannada safety audio.
2. MODERATE distress speech input completes full advisory with empathetic framing and synthesizes spoken audio.
3. Plant-death speech input ('ನನ್ನ ಬೆಳೆ ಸತ್ತುಹೋಗುತ್ತಿದೆ') evaluates to NONE without false-positive human escalation.
4. API routes (/api/v1/advisory and /api/v1/advisory/audio) include the structured distress object.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.advisory.voice_bridge import process_voice_advisory
from model.tts.synthesizer import MockKannadaSynthesizer
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge
from model.distress import get_distress_detector, SAFETY_RESPONSE_KN
from api.app import create_app


class TestVoiceDistressIntegration(unittest.TestCase):
    """Integration test suite for voice bridge and API endpoints with distress detection."""

    def setUp(self) -> None:
        self.language_bridge = MockLanguageBridge()
        self.mock_tts = MockKannadaSynthesizer()
        self.detector = get_distress_detector()

        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            distress_detector=self.detector
        )
        self.dummy_audio_path = "dataset/samples/sample_kannada_query.wav"
        self.temp_dir = tempfile.TemporaryDirectory()

        # Initialize test client for API route verification
        self.app = create_app(
            advisory_engine=self.engine,
            tts_engine=self.mock_tts,
            config={"TESTING": True}
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_01_high_distress_voice_flow_fast_path_and_tts(self, mock_transcribe) -> None:
        """
        HIGH Distress Voice Test:
        ASR transcription produces high-risk text.
        Pipeline must return immediate Kannada safety response and generate spoken TTS audio.
        """
        mock_transcribe.return_value = {
            "text": "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು",
            "duration_seconds": 2.0,
            "processing_time_seconds": 0.2,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }
        out_wav = os.path.join(self.temp_dir.name, "safety_response.wav")

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            tts_engine=self.mock_tts,
            synthesize_audio=True,
            output_audio_path=out_wav
        )

        # Verify distress metadata
        self.assertIn("distress", result)
        self.assertEqual(result["distress"]["level"], "HIGH")
        self.assertTrue(result["distress"]["detected"])
        self.assertEqual(result["distress"]["priority"], "safety")

        # Verify text and TTS audio
        self.assertEqual(result["response"], SAFETY_RESPONSE_KN)
        self.assertTrue(result["audio"]["available"])
        self.assertTrue(os.path.exists(out_wav))
        self.assertEqual(len(result["retrieved_documents"]), 0)

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_02_moderate_distress_voice_flow_with_tts(self, mock_transcribe) -> None:
        """
        MODERATE Distress Voice Test:
        ASR transcription produces financial/crop distress.
        Pipeline must generate full advisory with empathy and synthesize spoken audio.
        """
        mock_transcribe.return_value = {
            "text": "ನನ್ನ ಬೆಳೆ ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ. ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ?",
            "duration_seconds": 3.0,
            "processing_time_seconds": 0.3,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }
        out_wav = os.path.join(self.temp_dir.name, "moderate_response.wav")

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            tts_engine=self.mock_tts,
            synthesize_audio=True,
            output_audio_path=out_wav
        )

        self.assertIn("distress", result)
        self.assertEqual(result["distress"]["level"], "MODERATE")
        self.assertTrue(result["distress"]["detected"])
        self.assertEqual(result["distress"]["priority"], "advisory")
        self.assertIn("ಧೈರ್ಯವಾಗಿರಿ", result["response"])
        self.assertTrue(result["audio"]["available"])
        self.assertTrue(os.path.exists(out_wav))

    @patch("model.advisory.voice_bridge.transcribe_audio")
    def test_03_plant_death_voice_flow_evaluates_to_none(self, mock_transcribe) -> None:
        """
        Plant Death Voice Test:
        ASR transcription contains plant dying phrase ('ನನ್ನ ಬೆಳೆ ಸತ್ತುಹೋಗುತ್ತಿದೆ').
        Must evaluate to NONE distress without false escalation.
        """
        mock_transcribe.return_value = {
            "text": "ನನ್ನ ಬೆಳೆ ಸತ್ತುಹೋಗುತ್ತಿದೆ, ಏನು ಮಾಡಬೇಕು?",
            "duration_seconds": 2.5,
            "processing_time_seconds": 0.3,
            "model": "vasista22/whisper-kannada-small",
            "device": "cpu"
        }

        result = process_voice_advisory(
            audio_path=self.dummy_audio_path,
            advisory_engine=self.engine,
            tts_engine=self.mock_tts,
            synthesize_audio=False
        )

        self.assertIn("distress", result)
        self.assertEqual(result["distress"]["level"], "NONE")
        self.assertFalse(result["distress"]["detected"])

    def test_04_api_advisory_endpoint_returns_distress_schema(self) -> None:
        """Verify POST /api/v1/advisory JSON payload contains the 'distress' block."""
        # 1. Normal query
        res = self.client.post("/api/v1/advisory", json={"query": "ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("distress", data)
        self.assertEqual(data["distress"]["level"], "NONE")
        self.assertFalse(data["distress"]["detected"])

        # 2. Moderate distress query
        res_mod = self.client.post("/api/v1/advisory", json={"query": "ನನ್ನ ಬೆಳೆ ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ. ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ?"})
        self.assertEqual(res_mod.status_code, 200)
        data_mod = res_mod.get_json()
        self.assertIn("distress", data_mod)
        self.assertEqual(data_mod["distress"]["level"], "MODERATE")
        self.assertTrue(data_mod["distress"]["detected"])

        # 3. High distress query
        res_high = self.client.post("/api/v1/advisory", json={"query": "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು"})
        self.assertEqual(res_high.status_code, 200)
        data_high = res_high.get_json()
        self.assertIn("distress", data_high)
        self.assertEqual(data_high["distress"]["level"], "HIGH")
        self.assertTrue(data_high["distress"]["detected"])
        self.assertEqual(data_high["distress"]["priority"], "safety")
        self.assertEqual(data_high["answer"], SAFETY_RESPONSE_KN)


if __name__ == "__main__":
    unittest.main()
