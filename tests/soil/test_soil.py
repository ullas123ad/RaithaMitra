"""
Unit tests for the RaithaMitra Karnataka Soil Health Context Module.
Validates dataset integrity, regional soil profiling, measured test integration,
unmeasured value safety, failure handling, and performance.
"""

import time
import unittest
from typing import Dict, Any

from model.location.service import LocationService
from model.soil.models import SoilContext
from model.soil.service import SoilService, SoilServiceError
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge


class TestKarnatakaSoilProfilesDataset(unittest.TestCase):
    """Validates physical dataset integrity and Karnataka coverage."""

    def setUp(self) -> None:
        self.service = SoilService()

    def test_1_dataset_loads_successfully(self) -> None:
        """TEST 1: Verify soil dataset loads without errors and covers key districts."""
        self.assertGreater(self.service.total_profiles_count, 15)
        self.assertIn("mandya", self.service._profiles)
        self.assertIn("udupi", self.service._profiles)
        self.assertIn("belagavi", self.service._profiles)
        self.assertIn("dharwad", self.service._profiles)

    def test_2_soil_profile_schema_completeness(self) -> None:
        """TEST 2: Verify all profiles have required fields and source citations."""
        for dist_key, profile in self.service._profiles.items():
            self.assertTrue(bool(profile.get("district")))
            self.assertTrue(bool(profile.get("agro_climatic_zone")))
            self.assertIsInstance(profile.get("dominant_soil_types"), list)
            self.assertTrue(len(profile["dominant_soil_types"]) > 0)
            self.assertTrue(bool(profile.get("soil_order")))
            self.assertTrue(bool(profile.get("typical_ph_range")))
            self.assertTrue(bool(profile.get("source_authority")))
            self.assertTrue(bool(profile.get("source_document")))
            self.assertEqual(profile.get("last_verified"), "2026-08-19")


class TestSoilContextRetrieval(unittest.TestCase):
    """Validates district-level soil profile resolution and regional distinctions."""

    def setUp(self) -> None:
        self.service = SoilService()
        self.loc_service = LocationService()

    def test_3_mandya_ragi_retrieval(self) -> None:
        """TEST 3: Verify Mandya resolves Southern Dry Zone red sandy loam."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        ctx = self.service.get_soil_context(location=loc, crop="ragi")

        self.assertTrue(ctx.available)
        self.assertFalse(ctx.is_measured_data)
        self.assertEqual(ctx.district, "Mandya")
        self.assertEqual(ctx.taluk, "Pandavapura")
        self.assertEqual(ctx.village, "Melukote")
        self.assertIn("Southern Dry Zone", ctx.agro_climatic_zone)
        self.assertIn("Red sandy loam", ctx.dominant_soil_types)
        self.assertEqual(ctx.soil_order, "Alfisols")
        # Ensure measured nutrient values are strictly None
        self.assertIsNone(ctx.ph)
        self.assertIsNone(ctx.nitrogen)
        self.assertIsNone(ctx.phosphorus)
        self.assertIsNone(ctx.potassium)

    def test_4_udupi_paddy_coastal_soil_retrieval(self) -> None:
        """TEST 4: Verify Udupi resolves Coastal Zone acidic laterite/alluvium soil."""
        loc = self.loc_service.get_location(district="Udupi", taluk="Kundapura", village="Basrur")
        ctx = self.service.get_soil_context(location=loc, crop="paddy")

        self.assertTrue(ctx.available)
        self.assertFalse(ctx.is_measured_data)
        self.assertEqual(ctx.district, "Udupi")
        self.assertIn("Coastal Zone", ctx.agro_climatic_zone)
        self.assertTrue(any("Laterite" in s or "Coastal alluvium" in s for s in ctx.dominant_soil_types))
        self.assertIn("4.8", ctx.typical_ph_range)
        # Distinct from Mandya
        self.assertNotEqual(ctx.typical_ph_range, "6.2 - 7.5 (Slightly acidic to neutral)")

    def test_5_belagavi_maize_northern_black_soil_retrieval(self) -> None:
        """TEST 5: Verify Belagavi resolves Northern Zone deep black Vertisols."""
        loc = self.loc_service.get_location(district="Belagavi", taluk="Gokak", village="Arbhavi")
        ctx = self.service.get_soil_context(location=loc, crop="maize")

        self.assertTrue(ctx.available)
        self.assertEqual(ctx.district, "Belagavi")
        self.assertIn("Vertisols", ctx.soil_order)
        self.assertTrue(any("black soil" in s.lower() for s in ctx.dominant_soil_types))
        self.assertIn("7.5", ctx.typical_ph_range)

    def test_6_dharwad_and_bengaluru_profiles(self) -> None:
        """TEST 6: Verify Dharwad and Bengaluru regional profiles resolve accurately."""
        ctx_dharwad = self.service.get_soil_context(district="Dharwad")
        self.assertTrue(ctx_dharwad.available)
        self.assertIn("Vertisols", ctx_dharwad.soil_order)

        ctx_blr = self.service.get_soil_context(district="Bengaluru Rural")
        self.assertTrue(ctx_blr.available)
        self.assertEqual(ctx_blr.soil_order, "Alfisols")


class TestSoilDataDistinctionAndSafety(unittest.TestCase):
    """Validates strict separation between regional profiles and measured lab tests."""

    def setUp(self) -> None:
        self.service = SoilService()

    def test_7_regional_profile_has_no_fabricated_measurements(self) -> None:
        """TEST 7: Verify regional profiles leave ph, N, P, K, EC strictly None."""
        ctx = self.service.get_soil_context(district="Mandya")
        self.assertFalse(ctx.is_measured_data)
        self.assertIsNone(ctx.ph)
        self.assertIsNone(ctx.nitrogen)
        self.assertIsNone(ctx.phosphorus)
        self.assertIsNone(ctx.potassium)
        self.assertIsNone(ctx.organic_carbon)
        self.assertIsNone(ctx.electrical_conductivity)

    def test_8_measured_lab_data_integration(self) -> None:
        """TEST 8: Verify actual lab test data sets is_measured_data=True and populates fields."""
        lab_test = {
            "ph": 6.8,
            "organic_carbon": 0.48,
            "nitrogen": 220.5,
            "phosphorus": 18.2,
            "potassium": 195.0,
            "electrical_conductivity": 0.35,
            "micronutrients": {"Zinc": "Deficient (0.45 ppm)", "Boron": "Sufficient (0.60 ppm)"}
        }
        ctx = self.service.get_soil_context(district="Mandya", measured_data=lab_test)

        self.assertTrue(ctx.available)
        self.assertTrue(ctx.is_measured_data)
        self.assertEqual(ctx.ph, 6.8)
        self.assertEqual(ctx.nitrogen, 220.5)
        self.assertEqual(ctx.phosphorus, 18.2)
        self.assertEqual(ctx.potassium, 195.0)
        self.assertEqual(ctx.organic_carbon, 0.48)
        self.assertEqual(ctx.electrical_conductivity, 0.35)
        self.assertEqual(ctx.micronutrients["Zinc"], "Deficient (0.45 ppm)")

    def test_9_format_soil_context_regional_vs_measured(self) -> None:
        """TEST 9: Verify format_soil_context differentiates regional vs measured text."""
        # Regional profile format
        reg_ctx = self.service.get_soil_context(district="Mandya")
        reg_text = self.service.format_soil_context(reg_ctx)
        self.assertIn("REGIONAL SOIL HEALTH PROFILE", reg_text)
        self.assertIn("This is regional soil classification data, NOT the farmer's specific field test", reg_text)
        self.assertIn("Do NOT state that the farmer's soil has specific measured N/P/K", reg_text)

        # Measured lab test format
        lab_ctx = self.service.get_soil_context(
            district="Mandya",
            measured_data={"ph": 6.5, "nitrogen": 210.0}
        )
        lab_text = self.service.format_soil_context(lab_ctx)
        self.assertIn("FIELD-MEASURED SOIL TEST RECORD", lab_text)
        self.assertIn("pH: 6.5", lab_text)
        self.assertIn("Available N: 210.0 kg/ha", lab_text)

    def test_10_missing_location_returns_unavailable_safely(self) -> None:
        """TEST 10: Verify missing or invalid location returns available=False gracefully."""
        ctx_empty = self.service.get_soil_context()
        self.assertFalse(ctx_empty.available)
        self.assertIn("Location or district information is required", ctx_empty.status_message)

        ctx_unknown = self.service.get_soil_context(district="NonExistentDistrict")
        self.assertFalse(ctx_unknown.available)
        self.assertIn("No authoritative soil profile found", ctx_unknown.status_message)

        # Formatter produces empty string on unavailable context
        self.assertEqual(self.service.format_soil_context(ctx_empty), "")


class TestAdvisoryEngineSoilIntegration(unittest.TestCase):
    """Validates end-to-end AdvisoryEngine integration with SoilContext."""

    def setUp(self) -> None:
        self.config = AdvisoryConfig(backend="mock", use_rag=True)
        self.engine = AdvisoryEngine(
            config=self.config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            soil_service=SoilService()
        )
        self.loc_service = LocationService()

    def test_11_advisory_engine_attaches_soil_context(self) -> None:
        """TEST 11: Verify AdvisoryEngine resolves and attaches soil context."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        res = self.engine.generate_advisory(
            query="ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಯಾವುದು ಹಾಕಬೇಕು?",
            source_language="kn",
            location=loc,
            crop="ragi"
        )
        self.assertIn("soil", res)
        self.assertIsNotNone(res["soil"])
        self.assertTrue(res["soil"]["available"])
        self.assertEqual(res["soil"]["district"], "Mandya")
        self.assertFalse(res["soil"]["is_measured_data"])
        self.assertIn("Red sandy loam", res["soil"]["dominant_soil_types"])

    def test_12_soil_failure_tolerance(self) -> None:
        """TEST 12: Verify pipeline functions normally when soil is unavailable."""
        # Query with no location -> soil context unavailable
        res = self.engine.generate_advisory(
            query="PM-KISAN ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.",
            source_language="kn"
        )
        self.assertIn("soil", res)
        self.assertIsNone(res["soil"])
        self.assertTrue(len(res["retrieved_schemes"]) > 0)
        self.assertIn("PM-KISAN", res["intermediate_response"])
        self.assertTrue("ಪಿಎಂ-ಕಿಸಾನ್" in res["response"] or "PM-KISAN" in res["response"])

    def test_13_performance_benchmark(self) -> None:
        """TEST 13: Verify SoilService initialization (<20ms) and lookup (<5ms)."""
        t0 = time.perf_counter()
        svc = SoilService()
        init_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(100):
            svc.get_soil_context(district="Mandya", crop="ragi")
        avg_lookup_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

        self.assertLess(init_ms, 20.0, f"SoilService init took {init_ms:.2f} ms (expected < 20 ms)")
        self.assertLess(avg_lookup_ms, 5.0, f"Average soil lookup took {avg_lookup_ms:.2f} ms (expected < 5 ms)")


if __name__ == "__main__":
    unittest.main()
