"""
RaithaMitra API Routes Definition
=================================
Exposes /api/v1/health, /api/v1/version, and /api/v1/advisory endpoints.
"""

import logging
from typing import Any, Dict, Optional
from flask import Blueprint, current_app, jsonify, request

from model.location.service import LocationNotFoundError
from model.location.models import LocationValidationError
from model.advisory.agriparam_engine import AdvisoryValidationError

logger = logging.getLogger(__name__)

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def _error_response(code: str, message: str, status_code: int = 400):
    """Formats a standard JSON error response."""
    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }), status_code


@api_v1.route("/health", methods=["GET"])
def health():
    """
    Lightweight health check endpoint.
    Verifies that the API service is operational without initializing heavy models.
    """
    return jsonify({
        "status": "ok",
        "service": "RaithaMitra"
    }), 200


@api_v1.route("/version", methods=["GET"])
def version():
    """
    Returns public version and commit metadata.
    """
    return jsonify({
        "service": "RaithaMitra",
        "version": "1.0.0",
        "commit": "ff9e04c",
        "status": "operational"
    }), 200


@api_v1.route("/advisory", methods=["POST"])
def get_advisory():
    """
    Primary agricultural advisory endpoint.
    Accepts farmer query, optional location and crop parameters,
    and returns grounded agricultural recommendations.
    """
    # 1. Request Body & JSON Validation
    if not request.is_json:
        return _error_response("VALIDATION_ERROR", "Request Content-Type must be application/json.", 400)

    try:
        data = request.get_json(silent=True)
    except Exception:
        return _error_response("VALIDATION_ERROR", "Malformed JSON in request body.", 400)

    if data is None or not isinstance(data, dict):
        return _error_response("VALIDATION_ERROR", "Request body must be a valid JSON object.", 400)

    # 2. Query Validation
    if "query" not in data:
        return _error_response("VALIDATION_ERROR", "Field 'query' is required.", 400)

    raw_query = data.get("query")
    if not isinstance(raw_query, str):
        return _error_response("VALIDATION_ERROR", "Field 'query' must be a string.", 400)

    query = raw_query.strip()
    if not query:
        return _error_response("VALIDATION_ERROR", "Farmer query cannot be empty or whitespace.", 400)

    # 3. Optional String Fields Validation
    optional_str_fields = ["district", "taluk", "village", "crop", "language"]
    for field in optional_str_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], str):
                return _error_response("VALIDATION_ERROR", f"Field '{field}' must be a string.", 400)

    district = data.get("district", "").strip() if data.get("district") else None
    taluk = data.get("taluk", "").strip() if data.get("taluk") else None
    village = data.get("village", "").strip() if data.get("village") else None
    crop = data.get("crop", "").strip() if data.get("crop") else None
    language = data.get("language", "kn").strip() if data.get("language") else "kn"

    # 4. Location Resolution
    location_service = current_app.config.get("LOCATION_SERVICE")
    location = None
    if district or taluk or village:
        try:
            if location_service:
                location = location_service.get_location(
                    district=district,
                    taluk=taluk,
                    village=village
                )
        except LocationNotFoundError as e:
            loc_label = " / ".join(filter(None, [district, taluk, village]))
            return _error_response("LOCATION_NOT_FOUND", f"Location '{loc_label}' not found in Karnataka directory.", 404)
        except LocationValidationError as e:
            return _error_response("VALIDATION_ERROR", f"Invalid location parameters: {str(e)}", 400)
        except Exception as e:
            logger.error("Unexpected error during location lookup: %s", e)
            return _error_response("LOCATION_NOT_FOUND", "Specified Karnataka location could not be resolved.", 404)

    # 5. Weather Resolution
    weather_service = current_app.config.get("WEATHER_SERVICE")
    weather = None
    if location and weather_service:
        try:
            weather = weather_service.get_weather(location, crop=crop)
        except Exception as e:
            logger.warning("Weather fetch encountered non-fatal error: %s", e)
            weather = None

    # 6. Advisory Engine Execution
    engine = current_app.config.get("ADVISORY_ENGINE")
    if not engine:
        return _error_response("SERVICE_UNAVAILABLE", "Advisory engine service is currently unavailable.", 503)

    try:
        result = engine.generate_advisory(
            query=query,
            source_language=language,
            location=location,
            weather=weather,
            crop=crop
        )
    except AdvisoryValidationError as e:
        return _error_response("VALIDATION_ERROR", str(e), 400)
    except Exception as e:
        logger.error("Unexpected exception during advisory generation: %s", e, exc_info=True)
        return _error_response("INTERNAL_ERROR", "An internal server error occurred while processing the agricultural advisory.", 500)

    # 7. Response Schema Assembly
    response_payload = {
        "success": True,
        "language": result.get("target_language", language),
        "canonical_crop": result.get("canonical_crop"),
        "answer": result.get("response", ""),
        "location": result.get("location"),
        "weather": result.get("weather"),
        "soil": result.get("soil"),
        "schemes": result.get("retrieved_schemes", []),
        "market": result.get("market"),
        "metadata": {
            "model": result.get("model"),
            "backend": result.get("backend"),
            "rag_enabled": result.get("rag_enabled"),
            "retrieved_documents_count": len(result.get("retrieved_documents", [])),
            "retrieval_time_seconds": result.get("retrieval_time_seconds", 0.0),
            "translation_in_time_seconds": result.get("translation_in_time_seconds", 0.0),
            "generation_time_seconds": result.get("generation_time_seconds", 0.0),
            "translation_out_time_seconds": result.get("translation_out_time_seconds", 0.0),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0)
        }
    }

    return jsonify(response_payload), 200


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
MAX_AUDIO_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


@api_v1.route("/advisory/audio", methods=["POST"])
def get_audio_advisory():
    """
    Audio advisory endpoint for farmer speech queries.
    Accepts multipart/form-data with 'audio' file and optional location fields.
    Transcribes via ASR and returns grounded agricultural advisory.
    """
    import os
    import tempfile
    from pathlib import Path
    from model.advisory.voice_bridge import process_voice_advisory, VoiceAdvisoryError

    # 1. File Upload Validation
    if "audio" not in request.files and "file" not in request.files:
        return _error_response("VALIDATION_ERROR", "Audio file is required in multipart form data (key: 'audio').", 400)

    audio_file = request.files.get("audio") or request.files.get("file")
    if not audio_file or not audio_file.filename:
        return _error_response("VALIDATION_ERROR", "Uploaded audio file is empty or missing filename.", 400)

    # 2. File Extension Validation
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        return _error_response("VALIDATION_ERROR", f"Unsupported audio format '{file_ext}'. Allowed: {sorted(list(ALLOWED_AUDIO_EXTENSIONS))}", 400)

    # 3. Read form parameters
    form_data = request.form
    district = form_data.get("district", "").strip() if form_data.get("district") else None
    taluk = form_data.get("taluk", "").strip() if form_data.get("taluk") else None
    village = form_data.get("village", "").strip() if form_data.get("village") else None
    crop = form_data.get("crop", "").strip() if form_data.get("crop") else None
    language = form_data.get("language", "kn").strip() if form_data.get("language") else "kn"

    # 4. Save to secure temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        audio_file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_AUDIO_SIZE_BYTES:
            return _error_response("PAYLOAD_TOO_LARGE", f"Audio file exceeds maximum size limit of 5 MB ({file_size} bytes).", 413)

        if file_size == 0:
            return _error_response("VALIDATION_ERROR", "Uploaded audio file contains 0 bytes.", 400)

        # 5. Process through Voice Bridge
        engine = current_app.config.get("ADVISORY_ENGINE")
        location_service = current_app.config.get("LOCATION_SERVICE")
        weather_service = current_app.config.get("WEATHER_SERVICE")

        if not engine:
            return _error_response("SERVICE_UNAVAILABLE", "Advisory engine service is currently unavailable.", 503)

        result = process_voice_advisory(
            audio_path=temp_path,
            advisory_engine=engine,
            location_service=location_service,
            weather_service=weather_service,
            district=district,
            taluk=taluk,
            village=village,
            crop=crop,
            language=language
        )

        response_payload = {
            "success": True,
            "language": result.get("target_language", language),
            "canonical_crop": result.get("canonical_crop"),
            "answer": result.get("response", ""),
            "asr": result.get("asr"),
            "location": result.get("location"),
            "weather": result.get("weather"),
            "soil": result.get("soil"),
            "schemes": result.get("retrieved_schemes", []),
            "market": result.get("market"),
            "metadata": {
                "model": result.get("model"),
                "backend": result.get("backend"),
                "rag_enabled": result.get("rag_enabled"),
                "retrieved_documents_count": len(result.get("retrieved_documents", [])),
                "voice_pipeline_total_time_seconds": result.get("voice_pipeline_total_time_seconds", 0.0),
                "retrieval_time_seconds": result.get("retrieval_time_seconds", 0.0),
                "translation_in_time_seconds": result.get("translation_in_time_seconds", 0.0),
                "generation_time_seconds": result.get("generation_time_seconds", 0.0),
                "translation_out_time_seconds": result.get("translation_out_time_seconds", 0.0),
                "processing_time_seconds": result.get("processing_time_seconds", 0.0)
            }
        }

        return jsonify(response_payload), 200

    except VoiceAdvisoryError as e:
        return _error_response("VOICE_PROCESSING_ERROR", str(e), 400)
    except Exception as e:
        logger.error("Unexpected error during voice advisory processing: %s", e, exc_info=True)
        return _error_response("INTERNAL_ERROR", "An internal error occurred while processing the audio advisory.", 500)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
