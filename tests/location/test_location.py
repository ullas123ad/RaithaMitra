"""
Unit Tests for RaithaMitra Karnataka Location Context Module.

Tests district, taluk, village lookup, coordinate precision, Kannada/native script matching,
validation bounds, LGD metadata preservation, and multi-region coverage.
"""

import unittest

from model.location import (
    LocationContext,
    LocationNotFoundError,
    LocationService,
    LocationValidationError,
)


class TestKarnatakaLocationService(unittest.TestCase):
    """Test suite for Karnataka LocationService and LocationContext."""

    def setUp(self) -> None:
        """Initialize LocationService before each test."""
        self.service = LocationService()

    def test_valid_district_lookup(self) -> None:
        """Verify lookup of a valid district returns district-level centroid."""
        loc = self.service.get_location(district="Mandya")
        self.assertEqual(loc.state, "Karnataka")
        self.assertEqual(loc.district, "Mandya")
        self.assertEqual(loc.district_kn, "ಮಂಡ್ಯ")
        self.assertIsNone(loc.taluk)
        self.assertIsNone(loc.village)
        self.assertTrue(12.0 <= loc.latitude <= 13.0)
        self.assertTrue(76.0 <= loc.longitude <= 77.5)
        self.assertIsNotNone(loc.lgd_code)

    def test_valid_district_and_taluk_lookup(self) -> None:
        """Verify lookup of district + taluk returns taluk coordinates."""
        loc = self.service.get_location(district="Mandya", taluk="Pandavapura")
        self.assertEqual(loc.district, "Mandya")
        self.assertEqual(loc.taluk, "Pandavapura")
        self.assertEqual(loc.taluk_kn, "ಪಾಂಡವಪುರ")
        self.assertIsNone(loc.village)
        self.assertEqual(round(loc.latitude, 2), 12.50)
        self.assertEqual(round(loc.longitude, 2), 76.67)

    def test_valid_village_lookup(self) -> None:
        """Verify lookup of district + taluk + village returns exact village coordinates."""
        loc = self.service.get_location(
            district="Mandya",
            taluk="Pandavapura",
            village="Melukote",
        )
        self.assertEqual(loc.district, "Mandya")
        self.assertEqual(loc.taluk, "Pandavapura")
        self.assertEqual(loc.village, "Melukote")
        self.assertEqual(loc.village_kn, "ಮೇಲುಕೋಟೆ")
        self.assertEqual(round(loc.latitude, 2), 12.66)
        self.assertEqual(round(loc.longitude, 2), 76.65)
        self.assertEqual(loc.lgd_code, "614301")

    def test_correct_coordinates_returned(self) -> None:
        """Verify coordinate accuracy for a known landmark location (Bengaluru North / Yelahanka)."""
        loc = self.service.get_location(
            district="Bengaluru Urban",
            taluk="Bangalore North",
            village="Yelahanka",
        )
        self.assertTrue(13.0 <= loc.latitude <= 13.2)
        self.assertTrue(77.5 <= loc.longitude <= 77.7)

    def test_english_case_and_whitespace_normalization(self) -> None:
        """Verify case-insensitive and whitespace-tolerant lookups."""
        loc1 = self.service.get_location(district="  bengaluru urban  ", taluk="bangalore north", village="  yelahanka ")
        loc2 = self.service.get_location(district="BENGALURU URBAN", taluk="BANGALORE NORTH", village="YELAHANKA")
        self.assertEqual(loc1.village, "Yelahanka")
        self.assertEqual(loc2.village, "Yelahanka")
        self.assertEqual(loc1.latitude, loc2.latitude)
        self.assertEqual(loc1.longitude, loc2.longitude)

    def test_kannada_native_lookup(self) -> None:
        """Verify lookup using native Kannada script for district, taluk, and village."""
        # 1. District in Kannada
        loc_dist = self.service.get_location(district="ಮಂಡ್ಯ")
        self.assertEqual(loc_dist.district, "Mandya")

        # 2. Taluk in Kannada
        loc_taluk = self.service.get_location(district="ಮಂಡ್ಯ", taluk="ಪಾಂಡವಪುರ")
        self.assertEqual(loc_taluk.taluk, "Pandavapura")

        # 3. Village in Kannada
        loc_vil = self.service.get_location(district="ಮಂಡ್ಯ", taluk="ಪಾಂಡವಪುರ", village="ಮೇಲುಕೋಟೆ")
        self.assertEqual(loc_vil.village, "Melukote")
        self.assertEqual(loc_vil.village_kn, "ಮೇಲುಕೋಟೆ")

    def test_invalid_district_raises_error(self) -> None:
        """Verify querying an invalid district raises LocationNotFoundError."""
        with self.assertRaises(LocationNotFoundError):
            self.service.get_location(district="Atlantis")

    def test_invalid_taluk_raises_error(self) -> None:
        """Verify querying an invalid taluk raises LocationNotFoundError."""
        with self.assertRaises(LocationNotFoundError):
            self.service.get_location(district="Mandya", taluk="NonExistentTaluk")

    def test_invalid_village_raises_error(self) -> None:
        """Verify querying an invalid village raises LocationNotFoundError."""
        with self.assertRaises(LocationNotFoundError):
            self.service.get_location(district="Mandya", taluk="Pandavapura", village="NonExistentVillage")

    def test_coordinate_validation_bounds(self) -> None:
        """Verify LocationValidationError is raised on out-of-range coordinates."""
        with self.assertRaises(LocationValidationError):
            LocationContext(district="Mandya", latitude=95.0, longitude=76.8)

        with self.assertRaises(LocationValidationError):
            LocationContext(district="Mandya", latitude=12.5, longitude=-185.0)

    def test_empty_district_raises_validation_error(self) -> None:
        """Verify empty district name is rejected."""
        with self.assertRaises(LocationValidationError):
            LocationContext(district="")

    def test_all_31_karnataka_districts_present(self) -> None:
        """Verify all 31 districts of Karnataka are indexed in the service."""
        districts = self.service.list_districts()
        self.assertEqual(len(districts), 31)
        district_names = {d["name"] for d in districts}

        # Check key representative districts across all regions
        self.assertIn("Bengaluru Urban", district_names)
        self.assertIn("Belagavi", district_names)
        self.assertIn("Kalaburagi", district_names)
        self.assertIn("Mysuru", district_names)
        self.assertIn("Udupi", district_names)
        self.assertIn("Ballari", district_names)
        self.assertIn("Vijayanagara", district_names)
        self.assertIn("Bidar", district_names)
        self.assertIn("Davanagere", district_names)

    def test_multi_region_coordinate_verification(self) -> None:
        """Verify realistic coordinates across 5 distinct Karnataka agro-climatic zones."""
        # 1. Bengaluru / Southern Dry Zone
        loc_blr = self.service.get_location(district="Bengaluru Urban", taluk="Bangalore North", village="Hesaraghatta")
        self.assertTrue(13.0 <= loc_blr.latitude <= 13.3 and 77.4 <= loc_blr.longitude <= 77.7)

        # 2. Northern Dry Zone (Belagavi / Gokak)
        loc_bgm = self.service.get_location(district="Belagavi", taluk="Gokak", village="Arbhavi")
        self.assertTrue(16.0 <= loc_bgm.latitude <= 16.4 and 74.7 <= loc_bgm.longitude <= 75.0)

        # 3. North Eastern Transition Zone (Bidar / Basavakalyan)
        loc_bdr = self.service.get_location(district="Bidar", taluk="Basavakalyan", village="Hulsoor")
        self.assertTrue(17.6 <= loc_bdr.latitude <= 18.0 and 76.8 <= loc_bdr.longitude <= 77.2)

        # 4. Coastal Zone (Udupi / Kundapura)
        loc_udp = self.service.get_location(district="Udupi", taluk="Kundapura", village="Basrur")
        self.assertTrue(13.5 <= loc_udp.latitude <= 13.8 and 74.5 <= loc_udp.longitude <= 74.9)

        # 5. Central Dry Zone (Chitradurga / Hiriyur)
        loc_cta = self.service.get_location(district="Chitradurga", taluk="Hiriyur", village="Aimangala")
        self.assertTrue(13.8 <= loc_cta.latitude <= 14.2 and 76.4 <= loc_cta.longitude <= 76.8)

    def test_deterministic_repeated_lookup(self) -> None:
        """Verify lookups are strictly deterministic across multiple invocations."""
        results = [
            self.service.get_location(district="Shivamogga", taluk="Sagara", village="Keladi")
            for _ in range(10)
        ]
        first = results[0]
        for res in results[1:]:
            self.assertEqual(res.latitude, first.latitude)
            self.assertEqual(res.longitude, first.longitude)
            self.assertEqual(res.lgd_code, first.lgd_code)

    def test_source_metadata_preserved(self) -> None:
        """Verify official source and versioning metadata are populated on all outputs."""
        loc = self.service.get_location(district="Dharwad", taluk="Hubballi", village="Navalgund")
        self.assertIn("Local Government Directory", loc.source)
        self.assertEqual(loc.source_version, "2026-LGD-KRN-v1.0")
        self.assertEqual(loc.last_updated, "2026-08")

    def test_free_form_search_location(self) -> None:
        """Verify search_location matches district, taluk, or village queries."""
        # Search by village name
        results = self.service.search_location("Melukote")
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0].village, "Melukote")
        self.assertEqual(results[0].district, "Mandya")

        # Search by Kannada taluk name
        results_kn = self.service.search_location("ಗೋಕಾಕ")
        self.assertTrue(len(results_kn) >= 1)
        self.assertEqual(results_kn[0].taluk, "Gokak")

    def test_hierarchy_labels(self) -> None:
        """Verify English and Kannada formatted hierarchy strings."""
        loc = self.service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        self.assertEqual(loc.hierarchy_label, "Melukote, Pandavapura, Mandya, Karnataka")
        self.assertEqual(loc.hierarchy_label_kn, "ಮೇಲುಕೋಟೆ, ಪಾಂಡವಪುರ, ಮಂಡ್ಯ, ಕರ್ನಾಟಕ")

    def test_list_taluks_and_villages(self) -> None:
        """Verify listing child taluks and villages for cascading UI/selection."""
        taluks = self.service.list_taluks("Belagavi")
        self.assertTrue(len(taluks) >= 3)
        taluk_names = {t["name"] for t in taluks}
        self.assertIn("Gokak", taluk_names)
        self.assertIn("Chikkodi", taluk_names)

        villages = self.service.list_villages("Belagavi", "Gokak")
        self.assertTrue(len(villages) >= 2)
        village_names = {v["name"] for v in villages}
        self.assertIn("Arbhavi", village_names)


if __name__ == "__main__":
    unittest.main()
