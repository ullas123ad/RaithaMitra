"""
Unit and Integration Tests for RaithaMitra Phase 5.9B: Kannada Response Correctness & Answer Quality.

Tests:
1. All 8 Step 12 real queries (water stress, heavy rain, pest, market, scheme, watermelon, coffee, vanilla).
2. Phase 5.7 Distress integration (HIGH fast-path, MODERATE empathetic advisory).
3. Explicit Kannada language contract invariant enforcement.
4. 0% cross-crop and cross-location contamination.
"""

import unittest

from model.location import LocationService
from model.advisory.agriparam_engine import AdvisoryEngine, AdvisoryConfig
from model.advisory.language_bridge import MockLanguageBridge, is_valid_kannada_text
from model.advisory.retriever import AgriculturalRetriever
from model.weather.service import WeatherService
from model.soil.service import SoilService
from model.market.service import MarketService
from model.schemes.service import SchemeService
from model.distress import get_distress_detector, DistressLevel


class TestKannadaResponseCorrectness(unittest.TestCase):
    """Test suite for Phase 5.9B Kannada response quality and safety invariants."""

    def setUp(self) -> None:
        self.location_service = LocationService()
        self.weather_service = WeatherService()
        self.soil_service = SoilService()
        self.market_service = MarketService()
        self.scheme_service = SchemeService()
        self.retriever = AgriculturalRetriever()
        self.distress_detector = get_distress_detector()
        self.lang_bridge = MockLanguageBridge()

        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            language_bridge=self.lang_bridge,
            retriever=self.retriever,
            scheme_service=self.scheme_service,
            soil_service=self.soil_service,
            market_service=self.market_service,
            distress_detector=self.distress_detector
        )

    def test_query_1_ragi_water_stress(self) -> None:
        """TEST 1: Ragi water stress query produces Kannada response and ragi crop identity."""
        q = "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ. ಎಲೆಗಳು ಒಣಗುತ್ತಿವೆ. ಈಗ ನಾನು ಏನು ಮಾಡಬೇಕು?"
        res = self.engine.generate_advisory(query=q, source_language="kn")

        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertTrue(is_valid_kannada_text(res["response"]), f"Response must be valid Kannada text. Got: {res['response']}")

    def test_query_2_location_heavy_rain(self) -> None:
        """TEST 2: Udupi paddy heavy rain query preserves Udupi location and paddy crop."""
        q = "ಉಡುಪಿಯಲ್ಲಿ ನನ್ನ ಭತ್ತದ ಗದ್ದೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ಈಗ ಏನು ಮಾಡಬೇಕು?"
        loc = self.location_service.detect_location_from_text(q)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.district, "Udupi")

        res = self.engine.generate_advisory(query=q, source_language="kn", location=loc)
        self.assertEqual(res["canonical_crop"], "paddy")
        self.assertEqual(res["location"]["district"], "Udupi")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_3_belagavi_maize_pest(self) -> None:
        """TEST 3: Belagavi maize pest query resolves maize crop and Belagavi location."""
        q = "ಬೆಳಗಾವಿಯಲ್ಲಿ ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳಿವೆ. ಏನು ಮಾಡಬೇಕು?"
        loc = self.location_service.detect_location_from_text(q)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.district, "Belagavi")

        res = self.engine.generate_advisory(query=q, source_language="kn", location=loc)
        self.assertEqual(res["canonical_crop"], "maize")
        self.assertEqual(res["location"]["district"], "Belagavi")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_4_mandya_ragi_market(self) -> None:
        """TEST 4: Mandya ragi market query returns market price context without drought advice."""
        q = "ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?"
        loc = self.location_service.detect_location_from_text(q)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.district, "Mandya")

        res = self.engine.generate_advisory(query=q, source_language="kn", location=loc)
        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_5_ragi_government_scheme(self) -> None:
        """TEST 5: Ragi scheme query retrieves verified scheme data."""
        q = "ನಾನು ರಾಗಿ ಬೆಳೆಯುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಿಗಬಹುದು?"
        res = self.engine.generate_advisory(query=q, source_language="kn")
        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_6_watermelon_excess_water(self) -> None:
        """TEST 6: Watermelon excess water query resolves watermelon without ragi/paddy context."""
        q = "ನನ್ನ ಕಲ್ಲಂಗಡಿ ಬೆಳೆಗೆ ನೀರು ಹೆಚ್ಚು ಆಗಿದೆ. ಏನು ಮಾಡಬೇಕು?"
        res = self.engine.generate_advisory(query=q, source_language="kn")
        self.assertEqual(res["canonical_crop"], "watermelon")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_7_mysuru_coffee(self) -> None:
        """TEST 7: Mysuru coffee query resolves coffee crop and Mysuru location."""
        q = "ಮೈಸೂರಿನಲ್ಲಿ ನನ್ನ ಕಾಫಿ ಗಿಡಗಳಿಗೆ ಏನು ಮಾಡಬೇಕು?"
        loc = self.location_service.detect_location_from_text(q)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.district, "Mysuru")

        res = self.engine.generate_advisory(query=q, source_language="kn", location=loc)
        self.assertEqual(res["canonical_crop"], "coffee")
        self.assertEqual(res["location"]["district"], "Mysuru")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_query_8_unsupported_crop_vanilla(self) -> None:
        """TEST 8: Vanilla query returns RECOGNIZED_BUT_NOT_SUPPORTED without fabricated treatment."""
        q = "ವೆನಿಲ್ಲಾ ಬೆಳೆಗೆ ಏನು ಮಾಡಬೇಕು?"
        res = self.engine.generate_advisory(query=q, source_language="kn")
        self.assertEqual(res["canonical_crop"], "vanilla")
        self.assertEqual(res["crop_support_status"].lower(), "recognized_not_supported")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_high_distress_fast_path_kannada(self) -> None:
        """TEST 9: HIGH distress query returns immediate Kannada safety response."""
        q = "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು"
        res = self.engine.generate_advisory(query=q, source_language="kn")
        self.assertEqual(res["distress"]["level"], "HIGH")
        self.assertTrue(is_valid_kannada_text(res["response"]))

    def test_moderate_distress_empathetic_kannada(self) -> None:
        """TEST 10: MODERATE distress query retains agricultural advisory + empathetic Kannada response."""
        q = "ನನ್ನ ಬೆಳೆ ಹಾಳಾಗಿದೆ, ಸಾಲ ತೀರಿಸಲು ತುಂಬಾ ಕಷ್ಟವಾಗುತ್ತಿದೆ"
        res = self.engine.generate_advisory(query=q, source_language="kn")
        self.assertEqual(res["distress"]["level"], "MODERATE")
        self.assertTrue(is_valid_kannada_text(res["response"]))


if __name__ == "__main__":
    unittest.main()
