"""
HTTP and Mock client implementations for fetching weather data from Open-Meteo.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Standard WMO Weather interpretation codes (WW)
WMO_CODE_MAP: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    62: "Moderate rain",
    63: "Moderate rain",
    64: "Heavy rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherClientError(Exception):
    """Raised when weather client fails to retrieve or parse weather data."""
    pass


class BaseWeatherClient(ABC):
    """Abstract base class for weather data clients."""

    @abstractmethod
    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch raw weather payload for given coordinates.

        Args:
            latitude: Latitude (-90 to +90).
            longitude: Longitude (-180 to +180).
            timeout: HTTP request timeout in seconds.

        Returns:
            Dictionary containing weather data.

        Raises:
            WeatherClientError: If network request or parsing fails.
        """
        pass


class OpenMeteoClient(BaseWeatherClient):
    """Production Open-Meteo REST API client (₹0 cost, no API key required)."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        """Initialize with optional requests.Session for connection pooling."""
        self.session = session or requests.Session()

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Fetch live weather and short-term forecast from Open-Meteo."""
        params = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "hourly": "precipitation,temperature_2m",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "Asia/Kolkata",
            "forecast_days": 4,
        }

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "RaithaMitra-CropAdvisory/1.0 (Academic Major Project; KSSEM Bangalore)"},
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise WeatherClientError(f"Unexpected JSON response type: {type(data)}")
            return data

        except requests.Timeout as e:
            logger.warning("Open-Meteo request timed out for (%.4f, %.4f): %s", latitude, longitude, e)
            raise WeatherClientError(f"Open-Meteo request timed out after {timeout}s") from e

        except requests.RequestException as e:
            logger.warning("Open-Meteo request failed for (%.4f, %.4f): %s", latitude, longitude, e)
            raise WeatherClientError(f"Open-Meteo network request failed: {e}") from e

        except ValueError as e:
            logger.warning("Open-Meteo returned invalid JSON: %s", e)
            raise WeatherClientError(f"Failed to parse Open-Meteo JSON response: {e}") from e


class MockWeatherClient(BaseWeatherClient):
    """Deterministic Mock Weather Client for offline unit and integration tests."""

    def __init__(
        self,
        mode: str = "normal",
        custom_payload: Optional[Dict[str, Any]] = None,
        should_fail: bool = False,
    ) -> None:
        """Initialize MockWeatherClient.

        Args:
            mode: Pre-configured weather scenario ('normal', 'heavy_rain', 'drought_heat', 'cloudy').
            custom_payload: Optional raw dictionary payload to return.
            should_fail: If True, simulates network/API failure by raising WeatherClientError.
        """
        self.mode = mode
        self.custom_payload = custom_payload
        self.should_fail = should_fail

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Return pre-configured mock weather payload."""
        if self.should_fail:
            raise WeatherClientError("Simulated weather service outage / network error.")

        if self.custom_payload is not None:
            return self.custom_payload

        # Standard deterministic presets based on mode
        if self.mode == "heavy_rain":
            return {
                "latitude": latitude,
                "longitude": longitude,
                "timezone": "Asia/Kolkata",
                "current": {
                    "time": "2026-08-19T10:00",
                    "temperature_2m": 22.4,
                    "relative_humidity_2m": 94,
                    "precipitation": 18.5,
                    "weather_code": 65,  # Heavy rain
                    "wind_speed_10m": 16.8,
                },
                "hourly": {
                    "time": [f"2026-08-19T{h:02d}:00" for h in range(24)],
                    "precipitation": [2.0] * 24,  # 48 mm in 24h
                    "temperature_2m": [22.0] * 24,
                },
                "daily": {
                    "time": ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"],
                    "precipitation_sum": [48.0, 32.0, 15.0, 5.0],
                    "temperature_2m_max": [25.0, 24.0, 26.0, 27.0],
                    "temperature_2m_min": [20.0, 19.5, 20.0, 21.0],
                    "weather_code": [65, 63, 61, 3],
                },
            }

        elif self.mode == "drought_heat":
            return {
                "latitude": latitude,
                "longitude": longitude,
                "timezone": "Asia/Kolkata",
                "current": {
                    "time": "2026-08-19T14:00",
                    "temperature_2m": 37.8,
                    "relative_humidity_2m": 32,
                    "precipitation": 0.0,
                    "weather_code": 0,  # Clear sky
                    "wind_speed_10m": 12.4,
                },
                "hourly": {
                    "time": [f"2026-08-19T{h:02d}:00" for h in range(24)],
                    "precipitation": [0.0] * 24,
                    "temperature_2m": [35.0] * 24,
                },
                "daily": {
                    "time": ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"],
                    "precipitation_sum": [0.0, 0.0, 0.0, 0.0],
                    "temperature_2m_max": [38.5, 39.0, 38.0, 37.5],
                    "temperature_2m_min": [24.0, 25.0, 24.5, 24.0],
                    "weather_code": [0, 0, 1, 0],
                },
            }

        # Default 'normal' mode
        return {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "Asia/Kolkata",
            "current": {
                "time": "2026-08-19T12:00",
                "temperature_2m": 27.5,
                "relative_humidity_2m": 68,
                "precipitation": 0.2,
                "weather_code": 2,  # Partly cloudy
                "wind_speed_10m": 9.5,
            },
            "hourly": {
                "time": [f"2026-08-19T{h:02d}:00" for h in range(24)],
                "precipitation": [0.1] * 24,  # 2.4 mm in 24h
                "temperature_2m": [26.0] * 24,
            },
            "daily": {
                "time": ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22"],
                "precipitation_sum": [2.4, 4.2, 1.8, 0.0],
                "temperature_2m_max": [29.0, 28.5, 30.0, 30.5],
                "temperature_2m_min": [21.0, 20.5, 21.0, 21.5],
                "weather_code": [2, 3, 1, 0],
            },
        }
