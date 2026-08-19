"""
End-to-End Model Orchestration & Integration Audit Tests
=========================================================
Validates complete multi-component pipeline, context isolation,
cross-contamination prevention, graceful failure tolerances,
short query handling, non-agri filtering, and unknown crop behavior.
"""

import time
import unittest
from typing import Dict, Any

from model.location.service import LocationService
from model.weather.service import WeatherService
from model.weather.client import MockWeatherClient
from model.soil.service import SoilService
from model.schemes.service import SchemeService
from model.market.service import MarketService
from model.market.client import MockMarketClient
from model.advisory.crop_identifier import resolve_canonical_crop
from model.advisory.retriever import AgriculturalRetriever
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
    AdvisoryValidationError
)
from model.advisory.language_bridge import MockLanguageBridge


class TestPipelineOrchestrationAndIntegration(unittest.TestCase):
    """Audits the full multi-module advisory pipeline and context boundaries."""

    def setUp(self) -> None:
        self.loc_service = LocationService()
        self.weather_service = WeatherService(client=MockWeatherClient())
        self.soil_service = SoilService()
        self.scheme_service = SchemeService()
        self.market_service = MarketService(client=MockMarketClient())
        self.language_bridge = MockLanguageBridge()
        self.retriever = AgriculturalRetriever()

        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            retriever=self.retriever,
            scheme_service=self.scheme_service,
            soil_service=self.soil_service,
            market_service=self.market_service
        )

    # -------------------------------------------------------------------------
    # 1. Pipeline Integrity & Location Consistency
    # -------------------------------------------------------------------------
    def test_1_location_coordinates_carry_to_weather(self) -> None:
        """Verify district/taluk/village coordinates pass consistently to weather service."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        self.assertIsNotNone(loc)
        self.assertEqual(loc.district, "Mandya")
        self.assertEqual(loc.taluk, "Pandavapura")
        self.assertEqual(loc.village, "Melukote")
        self.assertAlmostEqual(loc.latitude, 12.6625, places=3)
        self.assertAlmostEqual(loc.longitude, 76.6542, places=3)

        weather = self.weather_service.get_weather(loc, crop="ragi")
        self.assertTrue(weather.available)
        self.assertIsNotNone(weather.location)
        self.assertAlmostEqual(weather.location.latitude, 12.6625, places=3)
        self.assertAlmostEqual(weather.location.longitude, 76.6542, places=3)

    # -------------------------------------------------------------------------
    # 2. Strict Cross-Crop Contamination Prevention (Section 21)
    # -------------------------------------------------------------------------
    def test_2_cross_crop_contamination_audit(self) -> None:
        """Verify Ragi, Chilli, Onion, Maize, Paddy queries strictly retrieve their own crop knowledge."""
        test_queries = [
            ("ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?", "ragi", "ragi"),
            ("ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?", "chilli", "chilli"),
            ("ನನ್ನ ಈರುಳ್ಳಿ ಬೆಳೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ಏನು ಮಾಡಬೇಕು?", "onion", "onion"),
            ("ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳು ಕಾಣಿಸುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?", "maize", "maize"),
            ("ಭತ್ತದ ಬೆಳೆಗೆ ಯಾವ ಕೀಟನಾಶಕ ಬಳಸಬೇಕು?", "paddy", "paddy"),
        ]

        for query_kn, expected_crop, expected_topic_crop in test_queries:
            canonical = resolve_canonical_crop(query=query_kn)
            self.assertEqual(canonical, expected_crop, f"Failed canonical crop resolution for {query_kn}")

            intermediate = self.language_bridge.translate_to_advisory_lang(query_kn)
            docs = self.retriever.retrieve(intermediate, crop=canonical, top_k=3)
            self.assertTrue(len(docs) > 0, f"No documents retrieved for {canonical}")
            for doc in docs:
                crop_val = doc.get("crop") if isinstance(doc, dict) else getattr(doc, "crop", None)
                self.assertEqual(
                    crop_val, expected_topic_crop,
                    f"Cross-crop contamination: Query for {expected_crop} retrieved document for {crop_val}"
                )

    # -------------------------------------------------------------------------
    # 3. 6 Core E2E Scenarios (Section 20)
    # -------------------------------------------------------------------------
    def test_3_e2e_1_ragi_drought_scenario(self) -> None:
        """E2E 1: Ragi drought query with Mandya location context."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        weather = self.weather_service.get_weather(loc, crop="ragi")

        res = self.engine.generate_advisory(
            query="ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            source_language="kn",
            location=loc,
            weather=weather,
            crop="ragi"
        )

        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertIsNotNone(res["weather"])
        self.assertIsNotNone(res["soil"])
        self.assertGreater(len(res["retrieved_documents"]), 0)
        self.assertEqual(res["retrieved_documents"][0]["crop"], "ragi")
        self.assertIn("ರಾಗಿ", res["response"])

    def test_4_e2e_2_paddy_heavy_rain_scenario(self) -> None:
        """E2E 2: Paddy heavy rainfall query in coastal Udupi."""
        loc = self.loc_service.get_location(district="Udupi", taluk="Kundapura", village="Basrur")
        weather = self.weather_service.get_weather(loc, crop="paddy")

        res = self.engine.generate_advisory(
            query="ಉಡುಪಿಯಲ್ಲಿ ನನ್ನ ಭತ್ತದ ಗದ್ದೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ಏನು ಪರಿಶೀಲಿಸಬೇಕು?",
            source_language="kn",
            location=loc,
            weather=weather,
            crop="paddy"
        )

        self.assertEqual(res["canonical_crop"], "paddy")
        self.assertIsNotNone(res["location"])
        self.assertEqual(res["location"]["district"], "Udupi")
        self.assertIsNotNone(res["weather"])
        self.assertGreater(len(res["response"]), 0)

    def test_5_e2e_3_maize_pest_scenario(self) -> None:
        """E2E 3: Maize leaf holes (Fall Armyworm) in Belagavi."""
        loc = self.loc_service.get_location(district="Belagavi", taluk="Gokak", village="Arbhavi")
        res = self.engine.generate_advisory(
            query="ಬೆಳಗಾವಿಯಲ್ಲಿ ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳಿವೆ. ಏನು ಮಾಡಬೇಕು?",
            source_language="kn",
            location=loc,
            crop="maize"
        )

        self.assertEqual(res["canonical_crop"], "maize")
        self.assertGreater(len(res["retrieved_documents"]), 0)
        self.assertEqual(res["retrieved_documents"][0]["crop"], "maize")
        self.assertIn("ಮೆಕ್ಕೆಜೋಳ", res["response"])

    def test_6_e2e_4_market_query_scenario(self) -> None:
        """E2E 4: Market price query in Mandya."""
        loc = self.loc_service.get_location(district="Mandya")
        res = self.engine.generate_advisory(
            query="ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            source_language="kn",
            location=loc,
            crop="ragi"
        )

        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertIsNotNone(res["market"])
        self.assertTrue(res["market"]["available"])
        self.assertGreater(len(res["market"]["records"]), 0)
        self.assertEqual(res["market"]["records"][0]["modal_price"], 3200.0)

    def test_7_e2e_5_scheme_query_scenario(self) -> None:
        """E2E 5: Government schemes query for ragi farmer."""
        res = self.engine.generate_advisory(
            query="ನಾನು ರಾಗಿ ಬೆಳೆಯುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?",
            source_language="kn",
            crop="ragi"
        )

        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertGreater(len(res["retrieved_schemes"]), 0)
        self.assertTrue(
            "ಪಿಎಂ-ಕಿಸಾನ್" in res["response"] or "ಯೋಜನೆ" in res["response"]
        )

    def test_8_e2e_6_non_agricultural_scenario(self) -> None:
        """E2E 6: Non-agricultural laptop repair query."""
        res = self.engine.generate_advisory(
            query="ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?",
            source_language="kn"
        )

        self.assertIsNone(res["canonical_crop"])
        self.assertEqual(len(res["retrieved_documents"]), 0)
        self.assertIsNone(res["market"])
        self.assertIn("ಕೃಷಿ", res["response"])

    # -------------------------------------------------------------------------
    # 4. Graceful Single-Module Failure Tolerances (Sections 16-19)
    # -------------------------------------------------------------------------
    def test_9_weather_failure_tolerance(self) -> None:
        """Verify weather failure allows advisory generation to complete safely."""
        failing_weather = self.weather_service.get_weather(location=None)  # unavailable
        self.assertFalse(failing_weather.available)

        res = self.engine.generate_advisory(
            query="ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            source_language="kn",
            weather=failing_weather,
            crop="ragi"
        )
        self.assertIsNotNone(res["response"])
        self.assertEqual(res["weather"]["available"], False)

    def test_10_market_failure_tolerance(self) -> None:
        """Verify market API failure returns available=False safely without crashing."""
        failing_market_svc = MarketService(client=MockMarketClient(should_fail=True))
        engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=self.language_bridge,
            market_service=failing_market_svc
        )

        res = engine.generate_advisory(
            query="ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            source_language="kn",
            crop="ragi"
        )
        self.assertIsNotNone(res["market"])
        self.assertFalse(res["market"]["available"])
        self.assertGreater(len(res["response"]), 0)

    def test_11_soil_failure_tolerance(self) -> None:
        """Verify unmapped or missing location soil returns available=False safely."""
        res = self.engine.generate_advisory(
            query="ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಯಾವುದು ಹಾಕಬೇಕು?",
            source_language="kn",
            location=None
        )
        self.assertIsNone(res["soil"])
        self.assertGreater(len(res["response"]), 0)

    # -------------------------------------------------------------------------
    # 5. Short Query & Unknown Crop Handling (Sections 12 & 14)
    # -------------------------------------------------------------------------
    def test_12_empty_query_raises_validation_error(self) -> None:
        """Verify empty input raises AdvisoryValidationError."""
        with self.assertRaises(AdvisoryValidationError):
            self.engine.generate_advisory(query="", source_language="kn")

    def test_13_single_word_crop_asks_clarification(self) -> None:
        """Verify single-word input 'ರಾಗಿ' asks what issue the farmer is facing."""
        res = self.engine.generate_advisory(query="ರಾಗಿ", source_language="kn")
        self.assertIn("ಸಮಸ್ಯೆ", res["response"])

    def test_14_unknown_crop_indicates_unavailable(self) -> None:
        """Verify unsupported crop 'ಕೇಸರಿ' (Saffron) states knowledge is unavailable."""
        res = self.engine.generate_advisory(
            query="ನಾನು ಕೇಸರಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೇನೆ.",
            source_language="kn"
        )
        self.assertIn("ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ", res["response"])

    # -------------------------------------------------------------------------
    # 6. Performance Benchmark (Section 23)
    # -------------------------------------------------------------------------
    def test_15_orchestration_overhead_benchmark(self) -> None:
        """Verify total local orchestration overhead is under 50 ms."""
        loc = self.loc_service.get_location(district="Mandya")
        # Warmup execution
        self.engine.generate_advisory(
            query="ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            source_language="kn",
            location=loc,
            crop="ragi"
        )
        t0 = time.perf_counter()
        self.engine.generate_advisory(
            query="ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            source_language="kn",
            location=loc,
            crop="ragi"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0, f"Orchestration overhead took {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
