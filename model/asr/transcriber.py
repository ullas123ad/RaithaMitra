"""
Kannada ASR Transcriber Module for RaithaMitra.

Integrates Hugging Face speech models (Whisper / Wav2Vec2 CTC) for Kannada speech-to-text.
Implements reusable model loading, GPU/CPU auto-selection, and inference execution.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional
import numpy as np

# Windows DLL directory fix for PyTorch on Python 3.13
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_torch_lib = os.path.join(base_dir, ".venv", "Lib", "site-packages", "torch", "lib")
    if os.path.exists(venv_torch_lib):
        try:
            os.add_dll_directory(venv_torch_lib)
        except Exception:
            pass

HAS_TORCH = False
TORCH_IMPORT_ERROR = None

try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        AutoModelForCTC,
        pipeline
    )
    HAS_TORCH = True
except (ImportError, OSError) as e:
    TORCH_IMPORT_ERROR = str(e)
    HAS_TORCH = False

from model.asr.config import ASRConfig
from model.asr.audio import load_and_preprocess_audio, AudioProcessingError

logger = logging.getLogger("RaithaMitra.ASR")


class ASRError(Exception):
    """Exception raised for ASR loading or inference failures."""
    pass


class KannadaASR:
    """
    Reusable Kannada ASR Engine.
    Caches model, processor, and pipeline instances to prevent repeated reloading.
    """

    _instance: Optional["KannadaASR"] = None

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.processor = None
        self.model = None
        self.asr_pipeline = None
        self.model_type = None  # "whisper" or "ctc"
        self.device = self._resolve_device()
        self.is_loaded = False

    def _resolve_device(self) -> str:
        """Determines best available compute device (CUDA GPU vs CPU)."""
        if self.config.device:
            return self.config.device
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> None:
        """
        Loads processor and ASR model weights into memory.
        Uses primary model (vasista22/whisper-kannada-small) with fallback.
        """
        if self.is_loaded:
            return

        if not HAS_TORCH:
            err_msg = f"PyTorch and Transformers could not be initialized: {TORCH_IMPORT_ERROR}" if TORCH_IMPORT_ERROR else "PyTorch and Transformers are required for ASR inference."
            raise ASRError(err_msg)

        logger.info(f"Loading Kannada ASR model '{self.config.model_id}' on device '{self.device}'...")
        start_time = time.time()

        models_to_try = [self.config.model_id, self.config.fallback_model_id]
        last_exception = None

        for model_name in models_to_try:
            try:
                device_idx = 0 if self.device == "cuda" else -1

                # Try loading via Hugging Face pipeline for seamless speech seq2seq / CTC support
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device_idx,
                    model_kwargs={"cache_dir": self.config.cache_dir} if self.config.cache_dir else {}
                )
                self.model_type = "pipeline"
                self.config.model_id = model_name
                self.is_loaded = True
                elapsed = time.time() - start_time
                logger.info(f"Successfully loaded '{model_name}' pipeline in {elapsed:.2f}s on {self.device}.")
                return

            except Exception as e:
                logger.warning(f"Pipeline load failed for '{model_name}': {e}. Trying direct model load...")
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=self.config.cache_dir)
                    try:
                        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=self.config.cache_dir)
                        self.model_type = "seq2seq"
                    except Exception:
                        self.model = AutoModelForCTC.from_pretrained(model_name, cache_dir=self.config.cache_dir)
                        self.model_type = "ctc"

                    self.model.to(self.device)
                    self.model.eval()

                    self.config.model_id = model_name
                    self.is_loaded = True
                    elapsed = time.time() - start_time
                    logger.info(f"Successfully loaded '{model_name}' ({self.model_type}) in {elapsed:.2f}s on {self.device}.")
                    return
                except Exception as direct_e:
                    logger.warning(f"Direct model load failed for '{model_name}': {direct_e}")
                    last_exception = direct_e

        raise ASRError(
            f"Failed to load any Kannada ASR model ({models_to_try}). Error details: {last_exception}"
        )

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes input Kannada audio file into Kannada text.

        Args:
            audio_path: Path to target audio file.

        Returns:
            Dict containing transcription text, language, model_id, duration, and metrics.
        """
        # 1. Preprocess and validate audio file first
        speech_array, audio_duration = load_and_preprocess_audio(
            audio_path=audio_path,
            target_sr=self.config.target_sampling_rate,
            max_duration_seconds=self.config.max_audio_duration_seconds
        )

        # 2. Ensure model is loaded into memory
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # 3. Run inference
        try:
            with torch.inference_mode():
                if self.model_type == "pipeline" and self.asr_pipeline is not None:
                    # Run Hugging Face pipeline inference (fine-tuned model naturally outputs Kannada)
                    gen_kwargs = {
                        "use_cache": getattr(self.config, "use_cache", True),
                        "num_beams": getattr(self.config, "num_beams", 1),
                        "do_sample": False,
                        "max_new_tokens": getattr(self.config, "max_new_tokens", 128)
                    }
                    pipeline_out = self.asr_pipeline(
                        {"raw": speech_array, "sampling_rate": self.config.target_sampling_rate},
                        generate_kwargs=gen_kwargs
                    )
                    if isinstance(pipeline_out, dict):
                        transcription_text = pipeline_out.get("text", "")
                    else:
                        transcription_text = str(pipeline_out)

                elif self.model_type == "seq2seq":
                    inputs = self.processor(
                        speech_array,
                        sampling_rate=self.config.target_sampling_rate,
                        return_tensors="pt"
                    )
                    input_features = inputs.input_features.to(self.device)
                    predicted_ids = self.model.generate(
                        input_features,
                        num_beams=getattr(self.config, "num_beams", 1),
                        do_sample=False,
                        use_cache=getattr(self.config, "use_cache", True),
                        max_new_tokens=getattr(self.config, "max_new_tokens", 128)
                    )
                    transcription_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                elif self.model_type == "ctc":
                    inputs = self.processor(
                        speech_array,
                        sampling_rate=self.config.target_sampling_rate,
                        return_tensors="pt",
                        padding=True
                    )
                    input_values = inputs.input_values.to(self.device)
                    logits = self.model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription_text = self.processor.batch_decode(predicted_ids)[0]
                else:
                    raise ASRError("ASR Model is not loaded properly.")

            # Clean whitespace
            transcription_text = " ".join(transcription_text.strip().split())

        except Exception as e:
            raise ASRError(f"Error during ASR model inference: {str(e)}")

        processing_time = time.time() - start_time

        return {
            "text": transcription_text,
            "language": self.config.expected_language,
            "model": self.config.model_id,
            "duration_seconds": round(audio_duration, 2),
            "processing_time_seconds": round(processing_time, 3),
            "device": self.device
        }


# Global singleton instance for pipeline reusability
_ASR_ENGINE: Optional[KannadaASR] = None


def get_asr_engine(config: Optional[ASRConfig] = None) -> KannadaASR:
    """Returns or initializes singleton ASR engine instance."""
    global _ASR_ENGINE
    if _ASR_ENGINE is None:
        _ASR_ENGINE = KannadaASR(config=config)
    return _ASR_ENGINE


def transcribe_audio(audio_path: str, config: Optional[ASRConfig] = None) -> Dict[str, Any]:
    """
    Public API function to transcribe an audio file into Kannada text.

    Usage:
        from model.asr import transcribe_audio
        result = transcribe_audio("dataset/samples/sample1.wav")
        print(result["text"])
    """
    engine = get_asr_engine(config=config)
    return engine.transcribe(audio_path)
