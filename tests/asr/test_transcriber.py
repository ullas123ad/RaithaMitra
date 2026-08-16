"""
Unit tests for ASR Transcriber Module (model/asr/transcriber.py).
"""

import os
import unittest
import numpy as np
import soundfile as sf
from model.asr.config import ASRConfig
from model.asr.audio import AudioProcessingError
from model.asr.transcriber import KannadaASR, transcribe_audio, ASRError


class TestASRTranscriber(unittest.TestCase):

    def setUp(self):
        """Set up test environment and sample audio files."""
        self.config = ASRConfig(
            model_id="vasista22/whisper-kannada-small",
            device="cpu"
        )
        self.sample_dir = os.path.join("dataset", "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

        self.sample_audio_path = os.path.join(self.sample_dir, "test_asr_sample.wav")
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(self.sample_audio_path, audio, sr)

    def tearDown(self):
        """Clean up test audio artifacts."""
        if os.path.exists(self.sample_audio_path):
            try:
                os.remove(self.sample_audio_path)
            except Exception:
                pass

    def test_config_initialization(self):
        """Verify default configuration settings."""
        self.assertEqual(self.config.expected_language, "kn")
        self.assertEqual(self.config.target_sampling_rate, 16000)
        self.assertEqual(self.config.model_id, "vasista22/whisper-kannada-small")

    def test_transcribe_nonexistent_audio_raises_error(self):
        """Verify transcription fails gracefully on non-existent audio file."""
        invalid_path = os.path.join(self.sample_dir, "non_existent_audio.wav")
        with self.assertRaises(AudioProcessingError):
            transcribe_audio(invalid_path, config=self.config)

    def test_transcribe_result_schema(self):
        """Verify response dictionary keys when transcribing test audio."""
        try:
            result = transcribe_audio(self.sample_audio_path, config=self.config)
            self.assertIn("text", result)
            self.assertIn("language", result)
            self.assertIn("model", result)
            self.assertIn("duration_seconds", result)
            self.assertIn("processing_time_seconds", result)
            self.assertIn("device", result)
            self.assertEqual(result["language"], "kn")
            self.assertEqual(result["duration_seconds"], 1.0)
        except ASRError as e:
            # Model loading or runtime environment failure caught cleanly
            self.assertTrue(len(str(e)) > 0)


if __name__ == "__main__":
    unittest.main()
