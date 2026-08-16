"""
Unit tests for Audio Preprocessing Module (model/asr/audio.py).
"""

import os
import unittest
import numpy as np
import soundfile as sf
from model.asr.audio import load_and_preprocess_audio, AudioProcessingError


class TestAudioPreprocessing(unittest.TestCase):

    def setUp(self):
        """Create temporary test audio files."""
        self.test_dir = os.path.join("dataset", "samples")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Valid 16kHz mono audio (1 second sine wave)
        self.valid_mono_path = os.path.join(self.test_dir, "test_synth_mono.wav")
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio_mono = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(self.valid_mono_path, audio_mono, sr)

        # Valid 44.1kHz stereo audio (1 second sine wave)
        self.valid_stereo_path = os.path.join(self.test_dir, "test_synth_stereo.wav")
        sr_stereo = 44100
        t_s = np.linspace(0, 1.0, sr_stereo, endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 440 * t_s).astype(np.float32)
        right = 0.4 * np.cos(2 * np.pi * 880 * t_s).astype(np.float32)
        audio_stereo = np.column_stack((left, right))
        sf.write(self.valid_stereo_path, audio_stereo, sr_stereo)

        # Empty 0-byte file
        self.empty_file_path = os.path.join(self.test_dir, "empty_test.wav")
        with open(self.empty_file_path, "wb") as f:
            pass

        # Corrupted invalid audio file
        self.corrupt_file_path = os.path.join(self.test_dir, "corrupt_test.wav")
        with open(self.corrupt_file_path, "wb") as f:
            f.write(b"NOT_A_REAL_WAV_HEADER_DATA_CONTENT_INVALID")

    def tearDown(self):
        """Clean up temporary test files."""
        for path in [self.valid_mono_path, self.valid_stereo_path, self.empty_file_path, self.corrupt_file_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_load_valid_mono_audio(self):
        """Verify valid mono audio loads and returns correct 16kHz float32 array."""
        speech_array, duration = load_and_preprocess_audio(self.valid_mono_path, target_sr=16000)
        self.assertIsInstance(speech_array, np.ndarray)
        self.assertEqual(speech_array.dtype, np.float32)
        self.assertEqual(speech_array.ndim, 1)
        self.assertAlmostEqual(duration, 1.0, places=1)
        self.assertEqual(len(speech_array), 16000)

    def test_load_valid_stereo_and_resample(self):
        """Verify stereo 44.1kHz audio converts to 16kHz mono."""
        speech_array, duration = load_and_preprocess_audio(self.valid_stereo_path, target_sr=16000)
        self.assertIsInstance(speech_array, np.ndarray)
        self.assertEqual(speech_array.ndim, 1)  # Mono
        self.assertEqual(len(speech_array), 16000)  # Resampled to 16kHz
        self.assertAlmostEqual(duration, 1.0, places=1)

    def test_nonexistent_file_raises_error(self):
        """Verify missing audio file raises AudioProcessingError."""
        missing_path = os.path.join(self.test_dir, "non_existent_file.wav")
        with self.assertRaises(AudioProcessingError):
            load_and_preprocess_audio(missing_path)

    def test_empty_file_raises_error(self):
        """Verify 0-byte audio file raises AudioProcessingError."""
        with self.assertRaises(AudioProcessingError):
            load_and_preprocess_audio(self.empty_file_path)

    def test_corrupted_file_raises_error(self):
        """Verify corrupted audio content raises AudioProcessingError."""
        with self.assertRaises(AudioProcessingError):
            load_and_preprocess_audio(self.corrupt_file_path)


if __name__ == "__main__":
    unittest.main()
