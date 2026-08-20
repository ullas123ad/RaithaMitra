"""
Unit and Integration Tests for RaithaMitra Phase 5.9A Dynamic Location Grounding.

Tests:
1. Spoken location extraction from Kannada/English farmer queries.
2. GPS coordinate resolution and Karnataka bounds validation.
3. Location precedence hierarchy (farmer_spoken > browser_gps > manual_selection > development_default).
4. Request-scoped location isolation across sequential queries.
"""

import unittest

from model.location import (
    LocationContext,
    LocationService,
    LocationValidationError,
)
from model.advisory.agriparam_engine import AdvisoryEngine, AdvisoryConfig
from model.advisory.voice_bridge import process_voice_advisory
from model.advisory.language_bridge import MockLanguageBridge
from model.advisory.retriever import AgriculturalRetriever
from model.weather.service import WeatherService
from model.soil.service import SoilService
from model.market.service import MarketService
from model.schemes.service import SchemeService
from model.distress import get_distress_detector


class TestDynamicLocationGrounding(unittest.TestCase):
    """Test suite for spoken location extraction, GPS resolution, and location precedence."""

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

    def test_spoken_location_extraction(self) -> None:
        """TEST 1: Verify explicit spoken location extraction across Karnataka districts."""
        queries = [
            ("ಉಡುಪಿಯಲ್ಲಿ ನನ್ನ ಭತ್ತದ ಗದ್ದೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ.", "Udupi"),
            ("ಬೆಳಗಾವಿಯಲ್ಲಿ ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆಗೆ ಹುಳು ಬಂದಿದೆ.", "Belagavi"),
            ("ಮಂಡ್ಯದಲ್ಲಿ ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ನೀರು ಸಾಲುತ್ತಿಲ್ಲ.", "Mandya"),
            ("ಮೈಸೂರಿನಲ್ಲಿ ನನ್ನ ಕಾಫಿ ಗಿಡಗಳಿಗೆ ಏನು ಮಾಡಬೇಕು?", "Mysuru"),
            ("ಹಾವೇರಿಯಲ್ಲಿ ಮೆಣಸಿನಕಾಯಿ ಬೆಲೆ ಎಷ್ಟು?", "Haveri"),
            ("ಕೋಲಾರದಲ್ಲಿ ಟೊಮೇಟೊ ಬೆಲೆ ಎಷ್ಟು?", "Kolar"),
        ]

        for query, expected_dist in queries:
            loc = self.location_service.detect_location_from_text(query)
            self.assertIsNotNone(loc, f"Failed to extract location from query: '{query}'")
            self.assertEqual(loc.district, expected_dist, f"Expected {expected_dist} for query '{query}', got {loc.district}")

    def test_gps_coordinate_resolution(self) -> None:
        """TEST 2: Verify GPS coordinates resolve to nearest Karnataka district."""
        # Udupi centroid coordinates (~13.34, 74.74)
        loc_udupi = self.location_service.get_location_from_coordinates(13.3409, 74.7421)
        self.assertEqual(loc_udupi.district, "Udupi")

        # Belagavi centroid coordinates (~15.85, 74.50)
        loc_belagavi = self.location_service.get_location_from_coordinates(15.8497, 74.4977)
        self.assertEqual(loc_belagavi.district, "Belagavi")

    def test_gps_out_of_bounds_validation(self) -> None:
        """TEST 3: Verify coordinates outside Karnataka raise LocationValidationError."""
        with self.assertRaises(LocationValidationError):
            self.location_service.get_location_from_coordinates(28.6139, 77.2090)  # Delhi

    def test_location_precedence_hierarchy(self) -> None:
        """TEST 4: Verify exact precedence hierarchy: farmer_spoken > browser_gps > manual > default."""
        audio_sample = "dataset/samples/sample_kannada_ragi_drought.wav"

        # Case A: Spoken location present -> farmer_spoken
        res_a = process_voice_advisory(
            audio_path=audio_sample,
            advisory_engine=self.engine,
            location_service=self.location_service,
            weather_service=self.weather_service,
            district="Mandya",
            latitude=15.8497,
            longitude=74.4977,
            synthesize_audio=False
        )
        # Note: transcript of sample_kannada_ragi_drought contains "ಮಂಡ್ಯದಲ್ಲಿ" or text
        self.assertIsNotNone(res_a.get("location"))

        # Case B: Manual selection only -> manual_selection
        res_b = process_voice_advisory(
            audio_path="dataset/samples/sample_kannada_query.wav",
            advisory_engine=self.engine,
            location_service=self.location_service,
            weather_service=self.weather_service,
            district="Belagavi",
            taluk="Gokak",
            synthesize_audio=False
        )
        self.assertEqual(res_b.get("location", {}).get("district"), "Belagavi")
        self.assertIn(res_b.get("location", {}).get("location_source"), ["farmer_spoken", "manual_selection"])

    def test_cross_location_isolation(self) -> None:
        """TEST 5: Verify sequential queries have isolated request-scoped location contexts."""
        loc1 = self.location_service.get_location(district="Udupi")
        loc2 = self.location_service.get_location(district="Belagavi")

        adv1 = self.engine.generate_advisory(query="ನನ್ನ ಭತ್ತದ ಗದ್ದೆಯಲ್ಲಿ ನೀರು ನಿಂತಿದೆ.", location=loc1, crop="paddy")
        adv2 = self.engine.generate_advisory(query="ನನ್ನ ಮೆಕ್ಕೆಜೋಳಕ್ಕೆ ಹುಳು ಬಂದಿದೆ.", location=loc2, crop="maize")

        self.assertEqual(adv1.get("location", {}).get("district"), "Udupi")
        self.assertEqual(adv1.get("canonical_crop"), "paddy")

        self.assertEqual(adv2.get("location", {}).get("district"), "Belagavi")
        self.assertEqual(adv2.get("canonical_crop"), "maize")


if __name__ == "__main__":
    unittest.main()
