"""
RaithaMitra Flask Application Factory
=====================================
Initializes the HTTP service, binds singleton model/data services,
configures CORS for frontend access, and registers error handlers.
"""

import logging
from typing import Any, Dict, Optional
from flask import Flask, jsonify
from flask_cors import CORS

from api.routes import api_v1

logger = logging.getLogger(__name__)


def create_app(
    advisory_engine: Optional[Any] = None,
    location_service: Optional[Any] = None,
    weather_service: Optional[Any] = None,
    soil_service: Optional[Any] = None,
    scheme_service: Optional[Any] = None,
    market_service: Optional[Any] = None,
    tts_engine: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Flask:
    """
    Creates and configures the RaithaMitra Flask application.

    Args:
        advisory_engine: Optional AdvisoryEngine instance (initialized lazily if omitted).
        location_service: Optional LocationService instance.
        weather_service: Optional WeatherService instance.
        soil_service: Optional SoilService instance.
        scheme_service: Optional SchemeService instance.
        market_service: Optional MarketService instance.
        tts_engine: Optional TTS synthesizer instance.
        config: Optional configuration dictionary.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    # Default application configurations
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max request body

    if config:
        app.config.update(config)

    # Configure CORS for local development and website consumption
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "Accept"],
        methods=["GET", "POST", "OPTIONS"]
    )

    # Initialize or assign singleton services
    if location_service is not None:
        app.config["LOCATION_SERVICE"] = location_service
    else:
        from model.location.service import LocationService
        app.config["LOCATION_SERVICE"] = LocationService()

    if weather_service is not None:
        app.config["WEATHER_SERVICE"] = weather_service
    else:
        from model.weather.service import WeatherService
        app.config["WEATHER_SERVICE"] = WeatherService()

    if soil_service is not None:
        app.config["SOIL_SERVICE"] = soil_service
    else:
        from model.soil.service import SoilService
        app.config["SOIL_SERVICE"] = SoilService()

    if scheme_service is not None:
        app.config["SCHEME_SERVICE"] = scheme_service
    else:
        from model.schemes.service import SchemeService
        app.config["SCHEME_SERVICE"] = SchemeService()

    if market_service is not None:
        app.config["MARKET_SERVICE"] = market_service
    else:
        from model.market.service import MarketService
        app.config["MARKET_SERVICE"] = MarketService()

    if advisory_engine is not None:
        app.config["ADVISORY_ENGINE"] = advisory_engine
    else:
        from model.advisory.agriparam_engine import AdvisoryEngine
        app.config["ADVISORY_ENGINE"] = AdvisoryEngine(
            scheme_service=app.config["SCHEME_SERVICE"],
            soil_service=app.config["SOIL_SERVICE"],
            market_service=app.config["MARKET_SERVICE"]
        )

    if tts_engine is not None:
        app.config["TTS_ENGINE"] = tts_engine
    else:
        from model.tts.synthesizer import get_tts_engine
        app.config["TTS_ENGINE"] = get_tts_engine()

    # Register Blueprints
    app.register_blueprint(api_v1)

    # Register Generic Error Handlers ensuring JSON responses
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Bad Request: " + str(error.description if hasattr(error, "description") else error)
            }
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": {
                "code": "RESOURCE_NOT_FOUND",
                "message": "The requested endpoint or resource was not found."
            }
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success": False,
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "HTTP method not allowed for this endpoint."
            }
        }), 405

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": {
                "code": "PAYLOAD_TOO_LARGE",
                "message": "Request payload exceeds maximum allowable size (2 MB)."
            }
        }), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred."
            }
        }), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
