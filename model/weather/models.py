"""
Data models for the RaithaMitra Karnataka-Specific Crop-Aware Weather Module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from model.location.models import LocationContext


@dataclass(frozen=True)
class CurrentWeather:
    """Current observed weather metrics."""

    temperature_c: float
    humidity_percent: float
    precipitation_mm: float
    wind_speed_kmh: float
    weather_condition: str
    weather_code: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ForecastWeather:
    """Short-term precipitation and temperature forecast metrics."""

    precipitation_next_24h_mm: float
    precipitation_next_3_days_mm: float
    temperature_max_next_24h_c: Optional[float] = None
    temperature_min_next_24h_c: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CropContext:
    """Normalized Karnataka crop identity."""

    requested: str
    canonical: str
    kannada_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class WeatherContext:
    """Comprehensive weather context model combining location, weather, and crop identity.

    Attributes:
        available: True if live/mock weather data was successfully fetched.
        location: The resolved Karnataka LocationContext.
        crop: Optional normalized CropContext.
        current: Optional CurrentWeather observation metrics.
        forecast: Optional ForecastWeather predictive metrics.
        observation_time: Timestamp of the weather observation from the provider.
        retrieved_at: ISO 8601 UTC timestamp when RaithaMitra fetched the record.
        source: Weather data provider (default 'Open-Meteo').
        status_message: Optional status or error message explaining availability state.
    """

    available: bool
    location: Optional[LocationContext] = None
    crop: Optional[CropContext] = None
    current: Optional[CurrentWeather] = None
    forecast: Optional[ForecastWeather] = None
    observation_time: Optional[str] = None
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "Open-Meteo"
    status_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested dataclasses serialized."""
        return {
            "available": self.available,
            "location": self.location.to_dict() if self.location else None,
            "crop": self.crop.to_dict() if self.crop else None,
            "current": self.current.to_dict() if self.current else None,
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "observation_time": self.observation_time,
            "retrieved_at": self.retrieved_at,
            "source": self.source,
            "status_message": self.status_message,
        }
