"""
Comprehensive Unit Tests for RaithaMitra Karnataka-Specific Crop-Aware Weather Module.

Tests weather parsing, current/forecast metrics, crop normalization, multi-region
Karnataka location integration, error handling, mock scenarios, timestamp preservation,
and graceful failure without internet.
"""

import unittest
from typing import Any, Dict

from model.advisory import AdvisoryConfig, AdvisoryEngine, MockAdvisoryBackend, MockLanguageBridge
from model.location import LocationContext, LocationService
from model.weather import (
    BaseWeatherClient,
    CropContext,
    CurrentWeather,
    ForecastWeather,
    MockWeatherClient,
    WeatherClientError,
    WeatherContext,
    WeatherService,
)


class TestKarnatakaWeatherModule(unittest.TestCase):
    """Test suite for WeatherService, MockWeatherClient, and Crop Normalization."""

    def setUp(self) -> None:
        """Initialize location and weather services with mock client."""
        self.location_service = LocationService()
        self.mock_client = MockWeatherClient(mode="normal")
        self.weather_service = WeatherService(client=self.mock_client)

        # Verified test locations from karnataka_locations.json
        self.loc_melukote = self.location_service.get_location(
            district="Mandya", taluk="Pandavapura", village="Melukote"
        )
        self.loc_arbhavi = self.location_service.get_location(
            district="Belagavi", taluk="Gokak", village="Arbhavi"
        )
        self.loc_basrur = self.location_service.get_location(
            district="Udupi", taluk="Kundapura", village="Basrur"
        )
        self.loc_aimangala = self.location_service.get_location(
            district="Chitradurga", taluk="Hiriyur", village="Aimangala"
        )
        self.loc_yelahanka = self.location_service.get_location(
            district="Bengaluru Urban", taluk="Bangalore North", village="Yelahanka"
        )

    def test_1_valid_weather_parsing(self) -> None:
        """TEST 1: Verify parsing of valid mock weather payload."""
        weather = self.weather_service.get_weather(self.loc_melukote, crop="ragi")
        self.assertTrue(weather.available)
        self.assertIsNotNone(weather.current)
        self.assertIsNotNone(weather.forecast)
        self.assertIsNotNone(weather.crop)
        self.assertEqual(weather.source, "Open-Meteo")
        self.assertEqual(weather.crop.canonical, "ragi")

    def test_2_current_weather_normalization(self) -> None:
        """TEST 2: Verify current weather metrics extraction and WMO code mapping."""
        weather = self.weather_service.get_weather(self.loc_melukote)
        cur = weather.current
        self.assertIsNotNone(cur)
        self.assertEqual(cur.temperature_c, 27.5)
        self.assertEqual(cur.humidity_percent, 68.0)
        self.assertEqual(cur.precipitation_mm, 0.2)
        self.assertEqual(cur.wind_speed_kmh, 9.5)
        self.assertEqual(cur.weather_code, 2)
        self.assertEqual(cur.weather_condition, "Partly cloudy")

    def test_3_forecast_normalization(self) -> None:
        """TEST 3: Verify 24h and 3-day precipitation forecast calculation."""
        weather = self.weather_service.get_weather(self.loc_melukote)
        fc = weather.forecast
        self.assertIsNotNone(fc)
        self.assertEqual(fc.precipitation_next_24h_mm, 2.4)
        self.assertEqual(fc.precipitation_next_3_days_mm, 8.4)
        self.assertEqual(fc.temperature_max_next_24h_c, 29.0)
        self.assertEqual(fc.temperature_min_next_24h_c, 21.0)

    def test_4_location_context_integration(self) -> None:
        """TEST 4: Verify LocationContext is faithfully attached to WeatherContext."""
        weather = self.weather_service.get_weather(self.loc_melukote)
        self.assertEqual(weather.location.district, "Mandya")
        self.assertEqual(weather.location.taluk, "Pandavapura")
        self.assertEqual(weather.location.village, "Melukote")
        self.assertEqual(weather.location.latitude, 12.6625)
        self.assertEqual(weather.location.longitude, 76.6542)

    def test_5_crop_normalization_ragi(self) -> None:
        """TEST 5: Verify Ragi crop normalization (English aliases and Kannada script)."""
        # English
        crop1 = self.weather_service.normalize_crop("finger millet")
        self.assertIsNotNone(crop1)
        self.assertEqual(crop1.canonical, "ragi")
        self.assertEqual(crop1.kannada_name, "ರಾಗಿ")

        # Kannada
        crop2 = self.weather_service.normalize_crop("ರಾಗಿ")
        self.assertIsNotNone(crop2)
        self.assertEqual(crop2.canonical, "ragi")

    def test_6_crop_normalization_paddy(self) -> None:
        """TEST 6: Verify Paddy / Rice crop normalization."""
        crop1 = self.weather_service.normalize_crop("rice")
        self.assertEqual(crop1.canonical, "paddy")
        self.assertEqual(crop1.kannada_name, "ಭತ್ತ")

        crop2 = self.weather_service.normalize_crop("ಭತ್ತ")
        self.assertEqual(crop2.canonical, "paddy")

    def test_7_crop_normalization_maize(self) -> None:
        """TEST 7: Verify Maize / Corn crop normalization."""
        crop1 = self.weather_service.normalize_crop("corn")
        self.assertEqual(crop1.canonical, "maize")
        self.assertEqual(crop1.kannada_name, "ಮೆಕ್ಕೆಜೋಳ")

        crop2 = self.weather_service.normalize_crop("ಮೆಕ್ಕೆಜೋಳ")
        self.assertEqual(crop2.canonical, "maize")

    def test_8_crop_normalization_groundnut(self) -> None:
        """TEST 8: Verify Groundnut / Peanut crop normalization."""
        crop1 = self.weather_service.normalize_crop("peanut")
        self.assertEqual(crop1.canonical, "groundnut")
        self.assertEqual(crop1.kannada_name, "ಕಡಲೆಕಾಯಿ")

        crop2 = self.weather_service.normalize_crop("ಕಡಲೆಕಾಯಿ")
        self.assertEqual(crop2.canonical, "groundnut")

    def test_9_crop_normalization_other_karnataka_crops(self) -> None:
        """TEST 9: Verify Sugarcane, Cotton, Chilli, Onion, Potato, Banana, Tomato normalization."""
        crops = [
            ("sugarcane", "sugarcane", "ಕಬ್ಬು"),
            ("cotton", "cotton", "ಹತ್ತಿ"),
            ("chilli", "chilli", "ಮೆಣಸಿನಕಾಯಿ"),
            ("onion", "onion", "ಈರುಳ್ಳಿ"),
            ("potato", "potato", "ಆಲೂಗಡ್ಡೆ"),
            ("banana", "banana", "ಬಾಳೆ"),
            ("tomato", "tomato", "ಟೊಮ್ಯಾಟೊ"),
        ]
        for query, expected_canonical, expected_kn in crops:
            norm = self.weather_service.normalize_crop(query)
            self.assertIsNotNone(norm)
            self.assertEqual(norm.canonical, expected_canonical)
            self.assertEqual(norm.kannada_name, expected_kn)

    def test_10_multi_region_karnataka_weather(self) -> None:
        """TEST 10: Verify weather context generation across 5 distinct Karnataka regions."""
        regions = [
            (self.loc_melukote, "ragi"),       # Southern Dry Zone
            (self.loc_arbhavi, "maize"),       # Northern Dry Zone
            (self.loc_basrur, "paddy"),        # Coastal Zone
            (self.loc_aimangala, "groundnut"), # Central Dry Zone
            (self.loc_yelahanka, "tomato"),    # Bengaluru Region
        ]
        for loc, crop in regions:
            weather = self.weather_service.get_weather(loc, crop=crop)
            self.assertTrue(weather.available)
            self.assertEqual(weather.location.district, loc.district)
            self.assertEqual(weather.crop.canonical, crop)
            self.assertIsNotNone(weather.current)

    def test_11_mock_heavy_rainfall_mode(self) -> None:
        """TEST 11: Verify parsing of heavy rainfall weather scenario."""
        client = MockWeatherClient(mode="heavy_rain")
        service = WeatherService(client=client)
        weather = service.get_weather(self.loc_basrur, crop="paddy")

        self.assertTrue(weather.available)
        self.assertEqual(weather.current.weather_condition, "Heavy rain")
        self.assertEqual(weather.current.precipitation_mm, 18.5)
        self.assertEqual(weather.forecast.precipitation_next_24h_mm, 48.0)
        self.assertEqual(weather.forecast.precipitation_next_3_days_mm, 95.0)

    def test_12_mock_drought_heat_mode(self) -> None:
        """TEST 12: Verify parsing of high temperature / drought scenario."""
        client = MockWeatherClient(mode="drought_heat")
        service = WeatherService(client=client)
        weather = service.get_weather(self.loc_arbhavi, crop="ragi")

        self.assertTrue(weather.available)
        self.assertEqual(weather.current.temperature_c, 37.8)
        self.assertEqual(weather.current.humidity_percent, 32.0)
        self.assertEqual(weather.forecast.precipitation_next_24h_mm, 0.0)

    def test_13_weather_failure_handled_gracefully(self) -> None:
        """TEST 13: Verify WeatherService handles simulated outages without crashing."""
        failing_client = MockWeatherClient(should_fail=True)
        service = WeatherService(client=failing_client)

        weather = service.get_weather(self.loc_melukote, crop="ragi")
        self.assertFalse(weather.available)
        self.assertIsNone(weather.current)
        self.assertIsNone(weather.forecast)
        self.assertIn("Simulated weather service outage", weather.status_message)
        self.assertEqual(weather.crop.canonical, "ragi")

    def test_14_missing_or_invalid_location_handled_safely(self) -> None:
        """TEST 14: Verify invalid location types return unavailable weather gracefully."""
        # Non-location object
        weather = self.weather_service.get_weather(location=None, crop="ragi")  # type: ignore
        self.assertFalse(weather.available)
        self.assertIn("Invalid LocationContext", weather.status_message)

    def test_15_timestamp_and_source_preservation(self) -> None:
        """TEST 15: Verify observation time, retrieved_at ISO timestamp, and source metadata."""
        weather = self.weather_service.get_weather(self.loc_melukote)
        self.assertEqual(weather.observation_time, "2026-08-19T12:00")
        self.assertTrue(weather.retrieved_at.startswith("2026-"))
        self.assertEqual(weather.source, "Open-Meteo")

    def test_16_format_weather_context_string(self) -> None:
        """TEST 16: Verify structured text formatting for Dhenu prompt context."""
        weather = self.weather_service.get_weather(self.loc_melukote, crop="ragi")
        formatted = self.weather_service.format_weather_context(weather)

        self.assertIn("--- LOCAL WEATHER CONTEXT (Open-Meteo) ---", formatted)
        self.assertIn("Location: Melukote, Pandavapura, Mandya, Karnataka", formatted)
        self.assertIn("Crop Context: Ragi (ರಾಗಿ)", formatted)
        self.assertIn("Condition: Partly cloudy", formatted)
        self.assertIn("Temperature: 27.5°C", formatted)
        self.assertIn("Expected Precipitation (Next 24h): 2.4 mm", formatted)
        self.assertIn("Total Expected Precipitation (Next 3 Days): 8.4 mm", formatted)

    def test_17_format_weather_context_when_unavailable(self) -> None:
        """TEST 17: Verify unavailable weather formats to empty string."""
        failing_client = MockWeatherClient(should_fail=True)
        service = WeatherService(client=failing_client)
        weather = service.get_weather(self.loc_melukote)
        formatted = service.format_weather_context(weather)
        self.assertEqual(formatted, "")

    def test_18_advisory_engine_end_to_end_with_weather(self) -> None:
        """TEST 18: Verify AdvisoryEngine generates structured output with weather attached."""
        config = AdvisoryConfig(backend="mock", use_rag=True)
        engine = AdvisoryEngine(
            config=config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge()
        )
        weather = self.weather_service.get_weather(self.loc_melukote, crop="ragi")

        result = engine.generate_advisory(
            query="ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಬರ ಬಂದಿದೆ, ಏನು ಮಾಡಬೇಕು?",
            source_language="kn",
            location=self.loc_melukote,
            weather=weather,
            crop="ragi"
        )

        self.assertIn("response", result)
        self.assertIsNotNone(result["weather"])
        self.assertTrue(result["weather"]["available"])
        self.assertEqual(result["weather"]["crop"]["canonical"], "ragi")
        self.assertEqual(result["location"]["village"], "Melukote")

    def test_19_advisory_engine_with_failed_weather_falls_back_cleanly(self) -> None:
        """TEST 19: Verify AdvisoryEngine still succeeds when weather service fails."""
        failing_client = MockWeatherClient(should_fail=True)
        weather_service = WeatherService(client=failing_client)
        weather = weather_service.get_weather(self.loc_melukote, crop="ragi")

        config = AdvisoryConfig(backend="mock", use_rag=True)
        engine = AdvisoryEngine(
            config=config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge()
        )

        result = engine.generate_advisory(
            query="There is drought and my ragi crop is drying",
            source_language="en",
            location=self.loc_melukote,
            weather=weather,
            crop="ragi"
        )

        self.assertIn("response", result)
        self.assertFalse(result["weather"]["available"])
        self.assertTrue(result["rag_enabled"])
        self.assertTrue(len(result["retrieved_documents"]) >= 1)

    def test_20_deterministic_repeated_weather_lookups(self) -> None:
        """TEST 20: Verify repeated lookups with mock client produce identical outputs."""
        w1 = self.weather_service.get_weather(self.loc_melukote, crop="ragi")
        w2 = self.weather_service.get_weather(self.loc_melukote, crop="ragi")
        self.assertEqual(w1.to_dict(), w2.to_dict())

    def test_21_to_dict_serialization(self) -> None:
        """TEST 21: Verify full JSON serializability of WeatherContext."""
        weather = self.weather_service.get_weather(self.loc_melukote, crop="ragi")
        d = weather.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["current"]["temperature_c"], 27.5)
        self.assertEqual(d["location"]["district"], "Mandya")
        self.assertEqual(d["crop"]["kannada_name"], "ರಾಗಿ")


if __name__ == "__main__":
    unittest.main()
