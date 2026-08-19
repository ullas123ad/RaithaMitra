"""
Comprehensive Unit and Integration Tests for RaithaMitra Backend API
====================================================================
Validates all endpoints, request validation, error shields, location resolution,
context attachment, graceful provider degradation, response schemas, and stack-trace safety.
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class TestRaithaMitraAPI(unittest.TestCase):
    """Test suite for RaithaMitra Flask API."""

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

    # -------------------------------------------------------------------------
    # 1. Health & Version Endpoints
    # -------------------------------------------------------------------------
    def test_1_health_endpoint(self) -> None:
        """TEST 1: Health check endpoint returns 200 OK and correct JSON."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "RaithaMitra")

    def test_2_version_endpoint(self) -> None:
        """TEST 2: Version endpoint returns operational metadata."""
        response = self.client.get("/api/v1/version")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("service"), "RaithaMitra")
        self.assertIn("version", data)
        self.assertIn("commit", data)

    # -------------------------------------------------------------------------
    # 2. Valid Advisory Requests & Language
    # -------------------------------------------------------------------------
    def test_3_valid_advisory_request(self) -> None:
        """TEST 3: Valid advisory request with Mandya location returns HTTP 200."""
        payload = {
            "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            "district": "Mandya",
            "taluk": "Pandavapura",
            "village": "Melukote",
            "crop": "ragi",
            "language": "kn"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("canonical_crop"), "ragi")
        self.assertIn("ರಾಗಿ", data.get("answer", ""))
        self.assertIsNotNone(data.get("location"))
        self.assertEqual(data["location"].get("district"), "Mandya")
        self.assertIsNotNone(data.get("weather"))
        self.assertIsNotNone(data.get("soil"))
        self.assertIn("metadata", data)
        self.assertGreater(data["metadata"].get("retrieved_documents_count", 0), 0)

    def test_4_english_query_request(self) -> None:
        """TEST 4: English language query request is processed cleanly."""
        payload = {
            "query": "What is the ragi price in Mandya?",
            "district": "Mandya",
            "language": "en"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("canonical_crop"), "ragi")
        self.assertIn("Mandya APMC", data.get("answer", ""))

    # -------------------------------------------------------------------------
    # 3. Request Validation & Error Handling
    # -------------------------------------------------------------------------
    def test_5_missing_query_field(self) -> None:
        """TEST 5: Request missing 'query' returns HTTP 400 with VALIDATION_ERROR."""
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"district": "Mandya"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("query", data["error"]["message"])

    def test_6_empty_query_field(self) -> None:
        """TEST 6: Request with empty query string returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"query": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_7_whitespace_query_field(self) -> None:
        """TEST 7: Request with whitespace-only query returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"query": "     "}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_8_malformed_json_body(self) -> None:
        """TEST 8: Non-JSON body or malformed JSON returns HTTP 400."""
        response = self.client.post(
            "/api/v1/advisory",
            data="{query: bad_json",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_9_wrong_field_types(self) -> None:
        """TEST 9: Wrong data type for query or district returns HTTP 400 without crashing."""
        # Non-string query
        resp1 = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"query": 12345}),
            content_type="application/json"
        )
        self.assertEqual(resp1.status_code, 400)
        self.assertEqual(resp1.get_json()["error"]["code"], "VALIDATION_ERROR")

        # Non-string district
        resp2 = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"query": "ರಾಗಿ ಬೆಳೆ", "district": ["Mandya"]}),
            content_type="application/json"
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertEqual(resp2.get_json()["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # 4. Crop Identification & Resolution
    # -------------------------------------------------------------------------
    def test_10_explicit_crop_parameter(self) -> None:
        """TEST 10: Explicit crop parameter is recognized and canonicalized."""
        payload = {
            "query": "ನನ್ನ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ",
            "crop": "Finger Millet"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("canonical_crop"), "ragi")

    def test_11_automatic_kannada_crop_detection(self) -> None:
        """TEST 11: Automatic Kannada crop detection (e.g. ಮೆಣಸಿನಕಾಯಿ -> chilli)."""
        payload = {
            "query": "ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("canonical_crop"), "chilli")

    # -------------------------------------------------------------------------
    # 5. Location Handling & Not-Found Status
    # -------------------------------------------------------------------------
    def test_12_invalid_location_not_found(self) -> None:
        """TEST 12: Non-existent Karnataka location returns HTTP 404 LOCATION_NOT_FOUND."""
        payload = {
            "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ",
            "district": "AtlantisDistrict"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "LOCATION_NOT_FOUND")
        self.assertIn("AtlantisDistrict", data["error"]["message"])

    # -------------------------------------------------------------------------
    # 6. Graceful External Provider Failure Degradation
    # -------------------------------------------------------------------------
    def test_13_weather_unavailable_graceful_handling(self) -> None:
        """TEST 13: Weather fetch failure does not crash API (weather.available = False)."""
        failing_weather_service = WeatherService(client=MockWeatherClient(should_fail=True))
        app = create_app(
            advisory_engine=self.engine,
            location_service=self.loc_service,
            weather_service=failing_weather_service,
            soil_service=self.soil_service,
            scheme_service=self.scheme_service,
            market_service=self.market_service,
            config={"TESTING": True}
        )
        client = app.test_client()

        payload = {
            "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            "district": "Mandya",
            "crop": "ragi"
        }
        response = client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        # Weather is cleanly marked unavailable or None, advisory still returned
        if data.get("weather"):
            self.assertFalse(data["weather"].get("available", True))

    def test_14_market_unavailable_graceful_handling(self) -> None:
        """TEST 14: Market failure returns available=False without fabricating price."""
        failing_market_svc = MarketService(client=MockMarketClient(should_fail=True))
        engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            scheme_service=self.scheme_service,
            soil_service=self.soil_service,
            market_service=failing_market_svc
        )
        app = create_app(
            advisory_engine=engine,
            location_service=self.loc_service,
            weather_service=self.weather_service,
            soil_service=self.soil_service,
            scheme_service=self.scheme_service,
            market_service=failing_market_svc,
            config={"TESTING": True}
        )
        client = app.test_client()

        payload = {
            "query": "ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            "district": "Mandya",
            "crop": "ragi"
        }
        response = client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsNotNone(data.get("market"))
        self.assertFalse(data["market"].get("available"))

    def test_15_soil_unavailable_graceful_handling(self) -> None:
        """TEST 15: Request without location returns soil as null or available=False."""
        payload = {
            "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಯಾವುದು ಹಾಕಬೇಕು?"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsNone(data.get("soil"))

    # -------------------------------------------------------------------------
    # 7. Non-Agricultural & Unsupported Crop Guardrails
    # -------------------------------------------------------------------------
    def test_16_non_agricultural_query(self) -> None:
        """TEST 16: Non-agricultural laptop repair query returns agricultural disclaimer."""
        payload = {
            "query": "ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsNone(data.get("canonical_crop"))
        self.assertEqual(len(data.get("schemes", [])), 0)
        self.assertIsNone(data.get("market"))
        self.assertIn("ಕೃಷಿ", data.get("answer", ""))

    def test_17_unsupported_crop_guidance(self) -> None:
        """TEST 17: Unsupported crop (Saffron / ಕೇಸರಿ) returns KVK referral notice."""
        payload = {
            "query": "ನಾನು ಕೇಸರಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೇನೆ."
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ", data.get("answer", ""))

    # -------------------------------------------------------------------------
    # 8. Schema Completeness & Security
    # -------------------------------------------------------------------------
    def test_18_response_json_structure_completeness(self) -> None:
        """TEST 18: Response contains all required top-level and metadata keys."""
        payload = {
            "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ",
            "district": "Mandya"
        }
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        required_keys = ["success", "language", "canonical_crop", "answer", "location", "weather", "soil", "schemes", "market", "metadata"]
        for k in required_keys:
            self.assertIn(k, data, f"Missing key '{k}' in response JSON")

        meta_keys = ["model", "backend", "rag_enabled", "retrieved_documents_count", "processing_time_seconds"]
        for mk in meta_keys:
            self.assertIn(mk, data["metadata"], f"Missing metadata key '{mk}'")

    def test_19_error_json_structure_and_no_stack_trace_leakage(self) -> None:
        """TEST 19: Error response does not leak Python exceptions or stack traces."""
        response = self.client.post(
            "/api/v1/advisory",
            data=json.dumps({"query": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()

        self.assertFalse(data.get("success"))
        self.assertIn("error", data)
        self.assertIn("code", data["error"])
        self.assertIn("message", data["error"])

        raw_text = response.get_data(as_text=True)
        self.assertNotIn("Traceback (most recent call last)", raw_text)
        self.assertNotIn("File \"", raw_text)

    def test_20_health_endpoint_is_lightweight(self) -> None:
        """TEST 20: Health endpoint returns in < 10 ms without loading heavy models."""
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            resp = self.client.get("/api/v1/health")
            self.assertEqual(resp.status_code, 200)
        avg_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
        self.assertLess(avg_ms, 10.0, f"Health endpoint average latency {avg_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
