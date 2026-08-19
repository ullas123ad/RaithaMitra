"""
Unit tests for the RaithaMitra Karnataka + Central Government Agricultural Scheme Module.
Validates dataset integrity, schema conformity, source traceability, alias resolution,
crop awareness, unverified scheme exclusion, and performance.
"""

import json
import os
import tempfile
import time
import unittest
from typing import Dict, Any

from model.location.service import LocationService
from model.schemes.models import (
    GovernmentScheme,
    VALID_VERIFICATION_STATUSES,
    ACTIVE_RECOMMENDED_STATUSES,
)
from model.schemes.service import SchemeService, SchemeServiceError
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge


class TestGovernmentSchemesDataset(unittest.TestCase):
    """Validates physical dataset integrity and strict source traceability."""

    def setUp(self) -> None:
        self.service = SchemeService()

    def test_1_dataset_loads_successfully(self) -> None:
        """TEST 1: Verify schemes dataset loads without errors."""
        self.assertGreater(self.service.total_count, 0)
        self.assertGreaterEqual(len(self.service.list_active_schemes()), 8)

    def test_2_schema_validation_all_records(self) -> None:
        """TEST 2: Verify all records have mandatory string and list fields."""
        for scheme in self.service._schemes.values():
            self.assertTrue(bool(scheme.id.strip()))
            self.assertTrue(bool(scheme.name_en.strip()))
            self.assertTrue(bool(scheme.name_kn.strip()))
            self.assertTrue(bool(scheme.government_level.strip()))
            self.assertTrue(bool(scheme.department.strip()))
            self.assertTrue(bool(scheme.category.strip()))
            self.assertTrue(bool(scheme.description.strip()))
            self.assertTrue(bool(scheme.purpose.strip()))
            self.assertIsInstance(scheme.eligible_farmer_types, list)
            self.assertTrue(bool(scheme.location_scope.strip()))
            self.assertTrue(bool(scheme.benefit_summary.strip()))
            self.assertTrue(bool(scheme.eligibility_summary.strip()))
            self.assertTrue(bool(scheme.application_method.strip()))

    def test_3_required_source_fields_present(self) -> None:
        """TEST 3: Verify source_authority, source_url, and source_document_title are populated."""
        for scheme in self.service._schemes.values():
            self.assertTrue(bool(scheme.source_authority.strip()))
            self.assertTrue(bool(scheme.source_url.strip()))
            self.assertTrue(scheme.source_url.startswith("http://") or scheme.source_url.startswith("https://"))
            self.assertTrue(bool(scheme.source_document_title.strip()))

    def test_4_required_verification_date(self) -> None:
        """TEST 4: Verify last_verified is current (2026-08-19) and valid status."""
        for scheme in self.service._schemes.values():
            self.assertEqual(scheme.last_verified, "2026-08-19")
            self.assertIn(scheme.verification_status, VALID_VERIFICATION_STATUSES)

    def test_5_active_scheme_filtering(self) -> None:
        """TEST 5: Verify list_active_schemes only returns ACTIVE_VERIFIED or state details."""
        active_list = self.service.list_active_schemes()
        self.assertTrue(len(active_list) > 0)
        for s in active_list:
            self.assertTrue(s.is_active)
            self.assertIn(s.verification_status, ACTIVE_RECOMMENDED_STATUSES)

    def test_6_unverified_scheme_exclusion(self) -> None:
        """TEST 6: Verify unverified/uncertain schemes are excluded from active list and searches."""
        test_scheme = self.service.get_scheme("unverified_test_pilot_scheme")
        self.assertIsNotNone(test_scheme)
        self.assertEqual(test_scheme.verification_status, "STATUS_UNCERTAIN")
        self.assertFalse(test_scheme.is_active)

        active_ids = [s.id for s in self.service.list_active_schemes()]
        self.assertNotIn("unverified_test_pilot_scheme", active_ids)

        # Search should never return unverified schemes
        results = self.service.search_schemes("solar drone pilot")
        result_ids = [s.id for s in results]
        self.assertNotIn("unverified_test_pilot_scheme", result_ids)


class TestSchemeRetrievalAndSearch(unittest.TestCase):
    """Validates search accuracy, canonical aliases, Kannada queries, and crop awareness."""

    def setUp(self) -> None:
        self.service = SchemeService()
        self.loc_service = LocationService()

    def test_7_pm_kisan_retrieval(self) -> None:
        """TEST 7: Verify PM-KISAN is accurately retrieved by English and Kannada queries."""
        # English query
        res_en = self.service.search_schemes("Tell me about PM-KISAN scheme")
        self.assertTrue(len(res_en) > 0)
        self.assertEqual(res_en[0].id, "pm_kisan")
        self.assertIn("6,000", res_en[0].benefit_summary)

        # Kannada query
        res_kn = self.service.search_schemes("PM-KISAN ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.")
        self.assertTrue(len(res_kn) > 0)
        self.assertEqual(res_kn[0].id, "pm_kisan")

    def test_8_pmfby_crop_insurance_retrieval(self) -> None:
        """TEST 8: Verify PMFBY crop insurance is retrieved by insurance queries."""
        # Kannada crop insurance query
        res_kn = self.service.search_schemes("ನನ್ನ ಬೆಳೆ ಹಾನಿಯಾಗಿದೆ. ಬೆಳೆ ವಿಮೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.")
        self.assertTrue(len(res_kn) > 0)
        self.assertEqual(res_kn[0].id, "pmfby_karnataka")
        self.assertIn("samrakshane", res_kn[0].application_portal)

        # English alias
        res_en = self.service.search_schemes("How to apply for PM Fasal Bima Yojana crop insurance?")
        self.assertTrue(len(res_en) > 0)
        self.assertEqual(res_en[0].id, "pmfby_karnataka")

    def test_9_kannada_general_scheme_query(self) -> None:
        """TEST 9: Verify general Kannada scheme query returns major active schemes."""
        res = self.service.search_schemes("ರೈತರಿಗೆ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು ಯಾವುವು?")
        self.assertTrue(len(res) >= 3)
        res_ids = [s.id for s in res]
        self.assertTrue("pm_kisan" in res_ids or "pmfby_karnataka" in res_ids or "karnataka_krishi_bhagya" in res_ids)

    def test_10_crop_aware_retrieval_ragi(self) -> None:
        """TEST 10: Verify Ragi crop query prioritizes Raita Siri millet scheme and PMFBY."""
        res = self.service.search_schemes(
            query="ನಾನು ರಾಗಿ ಬೆಳೆಸುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?",
            crop="ragi"
        )
        self.assertTrue(len(res) > 0)
        res_ids = [s.id for s in res]
        self.assertIn("karnataka_raita_siri", res_ids)

    def test_11_crop_aware_retrieval_horticulture(self) -> None:
        """TEST 11: Verify Banana/Tomato crop query prioritizes horticulture MIDH and micro-irrigation."""
        res = self.service.search_schemes(
            query="What government schemes or subsidies are available for banana orchard growers?",
            crop="banana"
        )
        self.assertTrue(len(res) > 0)
        res_ids = [s.id for s in res]
        self.assertTrue("midh_horticulture_karnataka" in res_ids or "pmksy_per_drop_more_crop" in res_ids)

    def test_12_karnataka_krishi_bhagya_retrieval(self) -> None:
        """TEST 12: Verify farm pond / rainwater harvesting query retrieves Krishi Bhagya."""
        res = self.service.search_schemes("ನನಗೆ ಕೃಷಿ ಹೊಂಡ ನಿರ್ಮಾಣಕ್ಕೆ ಸಹಾಯಧನ ಬೇಕು")
        self.assertTrue(len(res) > 0)
        self.assertEqual(res[0].id, "karnataka_krishi_bhagya")
        self.assertIn("Krishi Honda", res[0].benefit_summary)

    def test_13_kcc_crop_loan_retrieval(self) -> None:
        """TEST 13: Verify agricultural credit and interest subvention query retrieves KCC."""
        res = self.service.search_schemes("ಬೆಳೆ ಸಾಲ ಮತ್ತು ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಮಾಹಿತಿ")
        self.assertTrue(len(res) > 0)
        self.assertEqual(res[0].id, "kcc_credit_support")
        self.assertIn("3,00,000", res[0].benefit_summary)

    def test_14_unknown_scheme_handling_no_hallucination(self) -> None:
        """TEST 14: Verify unknown bogus scheme returns empty results."""
        res = self.service.search_schemes("XYZ ಕೃಷಿ ಯೋಜನೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ.")
        self.assertEqual(len(res), 0)

    def test_15_non_agricultural_query_rejection(self) -> None:
        """TEST 15: Verify non-agricultural queries return empty results."""
        res = self.service.search_schemes("ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?")
        self.assertEqual(len(res), 0)

        res2 = self.service.search_schemes("how to fix python coding bugs?")
        self.assertEqual(len(res2), 0)

    def test_16_empty_query_handled_safely(self) -> None:
        """TEST 16: Verify empty or whitespace query is handled safely."""
        self.assertEqual(self.service.search_schemes(""), [])
        self.assertEqual(self.service.search_schemes("   "), [])

    def test_17_format_scheme_context_output(self) -> None:
        """TEST 17: Verify format_scheme_context produces clean structured text."""
        schemes = [self.service.get_scheme("pm_kisan"), self.service.get_scheme("pmfby_karnataka")]
        text = self.service.format_scheme_context(schemes)
        self.assertIn("--- RELEVANT GOVERNMENT SCHEMES", text)
        self.assertIn("PM-KISAN", text)
        self.assertIn("PMFBY", text)
        self.assertIn("Important Rules for Scheme Guidance", text)


class TestSchemeDatasetValidationAndErrors(unittest.TestCase):
    """Validates strict error handling, duplicate ID rejection, and schema validation."""

    def test_18_duplicate_scheme_id_rejected(self) -> None:
        """TEST 18: Verify duplicate scheme IDs trigger SchemeServiceError."""
        dup_data = [
            {
                "id": "pm_kisan",
                "name_en": "PM-KISAN 1",
                "name_kn": "ಪಿಎಂ 1",
                "government_level": "Central",
                "department": "MoA",
                "category": "Income",
                "description": "Desc 1",
                "purpose": "Purpose 1",
                "eligible_farmer_types": ["All"],
                "eligible_crops": None,
                "location_scope": "All India",
                "benefit_summary": "₹6,000",
                "eligibility_summary": "Landholding",
                "application_method": "Online",
                "source_authority": "MoA",
                "source_document_title": "Doc 1",
                "source_url": "https://pmkisan.gov.in",
                "last_verified": "2026-08-19",
                "verification_status": "ACTIVE_VERIFIED"
            },
            {
                "id": "pm_kisan",
                "name_en": "PM-KISAN 2",
                "name_kn": "ಪಿಎಂ 2",
                "government_level": "Central",
                "department": "MoA",
                "category": "Income",
                "description": "Desc 2",
                "purpose": "Purpose 2",
                "eligible_farmer_types": ["All"],
                "eligible_crops": None,
                "location_scope": "All India",
                "benefit_summary": "₹6,000",
                "eligibility_summary": "Landholding",
                "application_method": "Online",
                "source_authority": "MoA",
                "source_document_title": "Doc 2",
                "source_url": "https://pmkisan.gov.in",
                "last_verified": "2026-08-19",
                "verification_status": "ACTIVE_VERIFIED"
            }
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(dup_data, f)
            temp_path = f.name

        try:
            with self.assertRaises(SchemeServiceError):
                SchemeService(data_path=temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_19_missing_required_source_url_rejected(self) -> None:
        """TEST 19: Verify missing source_url raises ValueError."""
        invalid_data = {
            "id": "test_invalid",
            "name_en": "Invalid Scheme",
            "name_kn": "ಅಮಾನ್ಯ ಯೋಜನೆ",
            "government_level": "Central",
            "department": "MoA",
            "category": "Test",
            "description": "Test Desc",
            "purpose": "Test Purpose",
            "eligible_farmer_types": ["All"],
            "eligible_crops": None,
            "location_scope": "Karnataka",
            "benefit_summary": "Benefit",
            "eligibility_summary": "Eligibility",
            "application_method": "Online",
            "source_authority": "Authority",
            "source_document_title": "Title",
            "source_url": "",  # EMPTY - INVALID
            "last_verified": "2026-08-19",
            "verification_status": "ACTIVE_VERIFIED"
        }
        with self.assertRaises(ValueError):
            GovernmentScheme.from_dict(invalid_data)


class TestAdvisoryEngineWithSchemes(unittest.TestCase):
    """Validates end-to-end AdvisoryEngine integration with SchemeService."""

    def setUp(self) -> None:
        self.config = AdvisoryConfig(
            model_id="mock",
            backend="mock",
            use_rag=True,
            advisory_language="en"
        )
        self.engine = AdvisoryEngine(
            config=self.config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            scheme_service=SchemeService()
        )
        self.loc_service = LocationService()

    def test_20_e2e_advisory_with_pm_kisan_query(self) -> None:
        """TEST 20: Verify AdvisoryEngine attaches retrieved PM-KISAN scheme context."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        result = self.engine.generate_advisory(
            query="PM-KISAN ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.",
            source_language="kn",
            location=loc
        )
        self.assertIn("retrieved_schemes", result)
        self.assertTrue(len(result["retrieved_schemes"]) > 0)
        top_scheme = result["retrieved_schemes"][0]
        self.assertEqual(top_scheme["id"], "pm_kisan")
        self.assertEqual(top_scheme["verification_status"], "ACTIVE_VERIFIED")
        self.assertEqual(top_scheme["source_authority"], "Ministry of Agriculture & Farmers Welfare, Government of India")

    def test_21_e2e_advisory_with_crop_aware_scheme_query(self) -> None:
        """TEST 21: Verify AdvisoryEngine attaches crop-specific schemes for Ragi query."""
        loc = self.loc_service.get_location(district="Mandya")
        result = self.engine.generate_advisory(
            query="ನಾನು ರಾಗಿ ಬೆಳೆಸುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?",
            source_language="kn",
            location=loc
        )
        self.assertIn("retrieved_schemes", result)
        self.assertTrue(len(result["retrieved_schemes"]) > 0)
        scheme_ids = [s["id"] for s in result["retrieved_schemes"]]
        self.assertIn("karnataka_raita_siri", scheme_ids)
        self.assertEqual(result["canonical_crop"], "ragi")

    def test_22_performance_benchmark(self) -> None:
        """TEST 22: Verify scheme loading (<50ms) and search (<20ms) performance."""
        # Load time test
        t0 = time.perf_counter()
        svc = SchemeService()
        load_time_ms = (time.perf_counter() - t0) * 1000.0

        # Search time test
        t0 = time.perf_counter()
        for _ in range(50):
            svc.search_schemes("PM-KISAN scheme and crop insurance in Karnataka", crop="ragi")
        avg_search_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

        self.assertLess(load_time_ms, 50.0, f"Scheme loading took {load_time_ms:.2f} ms (expected <50 ms)")
        self.assertLess(avg_search_ms, 20.0, f"Average search took {avg_search_ms:.2f} ms (expected <20 ms)")


if __name__ == "__main__":
    unittest.main()
