"""
Karnataka-Specific Crop-Aware Weather Service for RaithaMitra.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, List, Optional

from model.location.models import LocationContext, LocationValidationError
from model.weather.client import (
    BaseWeatherClient,
    OpenMeteoClient,
    WeatherClientError,
    WMO_CODE_MAP,
)
from model.weather.models import (
    CropContext,
    CurrentWeather,
    ForecastWeather,
    WeatherContext,
)

logger = logging.getLogger(__name__)

# Canonical mapping for common Karnataka agricultural crops
CROP_CANONICAL_MAP: Dict[str, Dict[str, str]] = {
    # Ragi / Finger Millet
    "ragi": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "finger millet": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "fingermillet": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "eleusine coracana": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "ರಾಗಿ": {"canonical": "ragi", "kannada": "ರಾಗಿ"},

    # Paddy / Rice
    "paddy": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "rice": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "paddy crop": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "bhatta": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "oryza sativa": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "ಭತ್ತ": {"canonical": "paddy", "kannada": "ಭತ್ತ"},

    # Maize / Corn
    "maize": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "corn": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "mekkejola": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "zea mays": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "ಮೆಕ್ಕೆಜೋಳ": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},

    # Groundnut / Peanut
    "groundnut": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "peanut": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "kadlekai": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "shenga": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "arachis hypogaea": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "ಕಡಲೆಕಾಯಿ": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},

    # Sugarcane
    "sugarcane": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "sugar cane": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "kabbu": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "saccharum officinarum": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "ಕಬ್ಬು": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},

    # Cotton
    "cotton": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "hatti": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "gossypium": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "ಹತ್ತಿ": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},

    # Chilli
    "chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "chili": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "green chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "red chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "menasinakai": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "capsicum": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "ಮೆಣಸಿನಕಾಯಿ": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},

    # Onion
    "onion": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "eerulli": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "allium cepa": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "ಈರುಳ್ಳಿ": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},

    # Potato
    "potato": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "aalugadde": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "solanum tuberosum": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "ಆಲೂಗಡ್ಡೆ": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},

    # Banana
    "banana": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "plantain": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "baale": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "musa": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "ಬಾಳೆ": {"canonical": "banana", "kannada": "ಬಾಳೆ"},

    # Tomato
    "tomato": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "tamota": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "solanum lycopersicum": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮ್ಯಾಟೊ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮೆಟೊ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
}


class WeatherService:
    """Service to fetch, parse, and structure Karnataka crop-aware weather contexts."""

    def __init__(self, client: Optional[BaseWeatherClient] = None) -> None:
        """Initialize WeatherService with live or mock weather client."""
        self.client = client or OpenMeteoClient()

    def normalize_crop(self, crop_name: Optional[str]) -> Optional[CropContext]:
        """Normalize an input crop name in English or Kannada to canonical representation."""
        if not crop_name or not isinstance(crop_name, str) or not crop_name.strip():
            return None

        # Clean and normalize text
        cleaned = unicodedata.normalize("NFC", crop_name.strip())
        lookup_key = " ".join(cleaned.lower().split())

        if lookup_key in CROP_CANONICAL_MAP:
            info = CROP_CANONICAL_MAP[lookup_key]
            return CropContext(
                requested=crop_name.strip(),
                canonical=info["canonical"],
                kannada_name=info["kannada"],
            )

        # Fallback for crops outside the predefined canonical set
        return CropContext(
            requested=crop_name.strip(),
            canonical=lookup_key,
            kannada_name=None,
        )

    def get_weather(
        self,
        location: LocationContext,
        crop: Optional[str] = None,
    ) -> WeatherContext:
        """Retrieve and structure weather context for a verified Karnataka location.

        Args:
            location: Validated Karnataka LocationContext.
            crop: Optional crop name in English or Kannada (e.g. 'ragi', 'ಭತ್ತ').

        Returns:
            Structured WeatherContext object.
        """
        crop_context = self.normalize_crop(crop)

        # Validate input location
        if not isinstance(location, LocationContext):
            logger.error("Invalid location object provided: %s", type(location))
            return WeatherContext(
                available=False,
                location=None,
                crop=crop_context,
                status_message="Invalid LocationContext object provided.",
            )

        lat = location.latitude
        lon = location.longitude

        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return WeatherContext(
                available=False,
                location=location,
                crop=crop_context,
                status_message=f"Coordinates ({lat}, {lon}) out of valid geographic range.",
            )

        try:
            raw_payload = self.client.fetch_weather(latitude=lat, longitude=lon)
        except WeatherClientError as e:
            logger.warning("Weather fetch failed for location '%s': %s", location.hierarchy_label, e)
            return WeatherContext(
                available=False,
                location=location,
                crop=crop_context,
                status_message=str(e),
            )
        except Exception as e:
            logger.error("Unexpected error in weather fetch: %s", e)
            return WeatherContext(
                available=False,
                location=location,
                crop=crop_context,
                status_message=f"Unexpected error: {e}",
            )

        # Parse current metrics
        current_data = raw_payload.get("current", {})
        obs_time = current_data.get("time")
        weather_code = int(current_data.get("weather_code", 0))
        weather_desc = WMO_CODE_MAP.get(weather_code, "Unknown")

        current_weather = CurrentWeather(
            temperature_c=float(current_data.get("temperature_2m", 0.0)),
            humidity_percent=float(current_data.get("relative_humidity_2m", 0.0)),
            precipitation_mm=float(current_data.get("precipitation", 0.0)),
            wind_speed_kmh=float(current_data.get("wind_speed_10m", 0.0)),
            weather_condition=weather_desc,
            weather_code=weather_code,
        )

        # Parse forecast metrics
        hourly_data = raw_payload.get("hourly", {})
        daily_data = raw_payload.get("daily", {})

        # 24h precipitation: sum of first 24 hourly entries or daily[0]
        hourly_precip = hourly_data.get("precipitation", [])
        if len(hourly_precip) >= 24:
            precip_24h = sum(float(p) for p in hourly_precip[:24])
        else:
            daily_precip = daily_data.get("precipitation_sum", [0.0])
            precip_24h = float(daily_precip[0]) if daily_precip else 0.0

        # 3-day precipitation: sum of daily precipitation_sum[0:3]
        daily_precip = daily_data.get("precipitation_sum", [])
        if len(daily_precip) >= 3:
            precip_3d = sum(float(p) for p in daily_precip[:3])
        elif len(hourly_precip) >= 72:
            precip_3d = sum(float(p) for p in hourly_precip[:72])
        else:
            precip_3d = precip_24h

        temp_max_daily = daily_data.get("temperature_2m_max", [])
        temp_min_daily = daily_data.get("temperature_2m_min", [])

        forecast_weather = ForecastWeather(
            precipitation_next_24h_mm=round(precip_24h, 1),
            precipitation_next_3_days_mm=round(precip_3d, 1),
            temperature_max_next_24h_c=float(temp_max_daily[0]) if temp_max_daily else None,
            temperature_min_next_24h_c=float(temp_min_daily[0]) if temp_min_daily else None,
        )

        return WeatherContext(
            available=True,
            location=location,
            crop=crop_context,
            current=current_weather,
            forecast=forecast_weather,
            observation_time=obs_time,
            source="Open-Meteo",
            status_message="Live weather retrieved successfully.",
        )

    def format_weather_context(self, weather: Optional[WeatherContext]) -> str:
        """Format WeatherContext into a clean, factual text block for the LLM prompt.

        Args:
            weather: WeatherContext instance.

        Returns:
            Structured text string.
        """
        if not weather or not weather.available:
            return ""

        lines: List[str] = ["--- LOCAL WEATHER CONTEXT (Open-Meteo) ---"]

        if weather.location:
            loc = weather.location
            lines.append(
                f"Location: {loc.hierarchy_label} ({loc.latitude:.4f}°N, {loc.longitude:.4f}°E)"
            )

        if weather.crop:
            crop = weather.crop
            kn_display = f" ({crop.kannada_name})" if crop.kannada_name else ""
            lines.append(f"Crop Context: {crop.canonical.capitalize()}{kn_display}")

        if weather.current:
            cur = weather.current
            obs_str = f" ({weather.observation_time})" if weather.observation_time else ""
            lines.append(f"Current Observation{obs_str}:")
            lines.append(f"  - Condition: {cur.weather_condition}")
            lines.append(f"  - Temperature: {cur.temperature_c:.1f}°C")
            lines.append(f"  - Relative Humidity: {cur.humidity_percent:.0f}%")
            lines.append(f"  - Precipitation: {cur.precipitation_mm:.1f} mm")
            lines.append(f"  - Wind Speed: {cur.wind_speed_kmh:.1f} km/h")

        if weather.forecast:
            fc = weather.forecast
            lines.append("Short-Term Forecast:")
            lines.append(f"  - Expected Precipitation (Next 24h): {fc.precipitation_next_24h_mm:.1f} mm")
            lines.append(f"  - Total Expected Precipitation (Next 3 Days): {fc.precipitation_next_3_days_mm:.1f} mm")
            if fc.temperature_max_next_24h_c is not None and fc.temperature_min_next_24h_c is not None:
                lines.append(
                    f"  - Temperature Range (Next 24h): {fc.temperature_min_next_24h_c:.1f}°C to {fc.temperature_max_next_24h_c:.1f}°C"
                )

        lines.append("Note: Weather is dynamic environmental context. Base crop protection and agronomic remedies strictly on verified practices in retrieved agricultural knowledge.")
        return "\n".join(lines)
