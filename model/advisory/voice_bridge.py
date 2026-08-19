"""
RaithaMitra Kannada Voice-to-Advisory Bridge
============================================
Connects the existing Kannada ASR transcription pipeline directly with
the multi-module AdvisoryEngine orchestration.
"""

import os
import time
import logging
from typing import Dict, Any, Optional

from model.asr.transcriber import transcribe_audio, ASRError
from model.asr.audio import AudioProcessingError
from model.advisory.agriparam_engine import AdvisoryEngine, AdvisoryValidationError

logger = logging.getLogger(__name__)


class VoiceAdvisoryError(Exception):
    """Exception raised for voice advisory pipeline failures."""
    pass


def process_voice_advisory(
    audio_path: str,
    advisory_engine: AdvisoryEngine,
    location_service: Optional[Any] = None,
    weather_service: Optional[Any] = None,
    district: Optional[str] = None,
    taluk: Optional[str] = None,
    village: Optional[str] = None,
    crop: Optional[str] = None,
    language: str = "kn",
    asr_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end Voice-to-Advisory execution:
    1. Transcribes input Kannada audio using existing ASR.
    2. Resolves optional location and weather contexts.
    3. Executes advisory pipeline (Crop Resolution -> Soil -> RAG -> LLM -> NLLB).
    4. Returns combined result with ASR transcription metadata.

    Args:
        audio_path: Absolute or relative path to the farmer's audio query file.
        advisory_engine: Initialized AdvisoryEngine instance.
        location_service: Optional LocationService for administrative resolution.
        weather_service: Optional WeatherService for live weather retrieval.
        district: Optional Karnataka district name.
        taluk: Optional Karnataka taluk name.
        village: Optional Karnataka village name.
        crop: Optional explicit crop override.
        language: Target output language ('kn' or 'en').
        asr_config: Optional ASRConfig overrides.

    Returns:
        Dict containing ASR transcription, advisory answer, contexts, and latency breakdown.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise VoiceAdvisoryError(f"Audio file not found: '{audio_path}'")

    t_total_start = time.time()

    # 1. Step 1: Speech-to-Text via existing Kannada ASR
    try:
        t_asr_start = time.time()
        asr_result = transcribe_audio(audio_path, config=asr_config)
        asr_time = time.time() - t_asr_start
    except (AudioProcessingError, ASRError) as e:
        raise VoiceAdvisoryError(f"ASR transcription failed: {str(e)}")
    except Exception as e:
        raise VoiceAdvisoryError(f"Unexpected error during speech transcription: {str(e)}")

    transcript = asr_result.get("text", "").strip()
    if not transcript:
        raise VoiceAdvisoryError("ASR produced an empty transcription from the provided audio.")

    # 2. Step 2: Location & Weather Context Resolution
    location = None
    if location_service and (district or taluk or village):
        try:
            location = location_service.get_location(
                district=district,
                taluk=taluk,
                village=village
            )
        except Exception as e:
            logger.warning("Location lookup failed in voice bridge: %s", e)
            location = None

    weather = None
    if weather_service and location:
        try:
            weather = weather_service.get_weather(location, crop=crop)
        except Exception as e:
            logger.warning("Weather lookup failed in voice bridge: %s", e)
            weather = None

    # 3. Step 3: Execute existing Advisory Orchestration
    try:
        t_adv_start = time.time()
        advisory_result = advisory_engine.generate_advisory(
            query=transcript,
            source_language=language,
            location=location,
            weather=weather,
            crop=crop
        )
        adv_time = time.time() - t_adv_start
    except AdvisoryValidationError as e:
        raise VoiceAdvisoryError(f"Advisory validation error on transcript: {str(e)}")
    except Exception as e:
        raise VoiceAdvisoryError(f"Advisory generation failed: {str(e)}")

    total_pipeline_time = time.time() - t_total_start

    # 4. Step 4: Attach ASR metadata to response
    advisory_result["asr"] = {
        "transcript": transcript,
        "audio_duration_seconds": asr_result.get("duration_seconds", 0.0),
        "asr_processing_time_seconds": round(asr_time, 4),
        "asr_model": asr_result.get("model", ""),
        "asr_device": asr_result.get("device", "")
    }
    advisory_result["voice_pipeline_total_time_seconds"] = round(total_pipeline_time, 4)

    return advisory_result
