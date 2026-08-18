"""
Data models for the RaithaMitra Karnataka Location Context Module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class LocationValidationError(ValueError):
    """Raised when location coordinates or fields fail validation."""
    pass


@dataclass(frozen=True)
class LocationContext:
    """Normalized representation of a verified Karnataka location.

    Attributes:
        state: State name (always 'Karnataka').
        state_kn: State name in Kannada ('ಕರ್ನಾಟಕ').
        district: English district name (e.g., 'Bengaluru Urban', 'Mandya').
        district_kn: Kannada district name (e.g., 'ಬೆಂಗಳೂರು ನಗರ', 'ಮಂಡ್ಯ').
        taluk: Optional English taluk name.
        taluk_kn: Optional Kannada taluk name.
        village: Optional English village name.
        village_kn: Optional Kannada village name.
        latitude: Latitude in decimal degrees (-90.0 to 90.0).
        longitude: Longitude in decimal degrees (-180.0 to 180.0).
        lgd_code: Optional Local Government Directory (LGD) code.
        source: Authoritative data source name.
        source_version: Source release/dataset version identifier.
        last_updated: Last verified/updated timestamp or string.
    """

    state: str = "Karnataka"
    state_kn: str = "ಕರ್ನಾಟಕ"
    district: str = ""
    district_kn: Optional[str] = None
    taluk: Optional[str] = None
    taluk_kn: Optional[str] = None
    village: Optional[str] = None
    village_kn: Optional[str] = None
    latitude: float = 0.0
    longitude: float = 0.0
    lgd_code: Optional[str] = None
    source: str = "Local Government Directory (LGD), Ministry of Panchayati Raj, Government of India & KSRSAC"
    source_version: str = "2026-LGD-KRN-v1.0"
    last_updated: str = "2026-08"

    def __post_init__(self) -> None:
        """Validate coordinates and required fields."""
        if not self.district or not isinstance(self.district, str) or not self.district.strip():
            raise LocationValidationError("District name is required and cannot be empty.")

        if not isinstance(self.latitude, (int, float)) or isinstance(self.latitude, bool):
            raise LocationValidationError(f"Invalid latitude type: {type(self.latitude)}")

        if not isinstance(self.longitude, (int, float)) or isinstance(self.longitude, bool):
            raise LocationValidationError(f"Invalid longitude type: {type(self.longitude)}")

        lat = float(self.latitude)
        lon = float(self.longitude)

        if not (-90.0 <= lat <= 90.0):
            raise LocationValidationError(f"Latitude {lat} out of valid range [-90, +90]")

        if not (-180.0 <= lon <= 180.0):
            raise LocationValidationError(f"Longitude {lon} out of valid range [-180, +180]")

    @property
    def hierarchy_label(self) -> str:
        """Return a human-readable English hierarchical label."""
        parts = []
        if self.village:
            parts.append(self.village)
        if self.taluk:
            parts.append(self.taluk)
        parts.append(self.district)
        parts.append(self.state)
        return ", ".join(parts)

    @property
    def hierarchy_label_kn(self) -> str:
        """Return a human-readable Kannada hierarchical label."""
        parts = []
        if self.village_kn:
            parts.append(self.village_kn)
        if self.taluk_kn:
            parts.append(self.taluk_kn)
        if self.district_kn:
            parts.append(self.district_kn)
        parts.append(self.state_kn)
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LocationContext:
        """Create a LocationContext instance from dictionary data."""
        return cls(
            state=data.get("state", "Karnataka"),
            state_kn=data.get("state_kn", "ಕರ್ನಾಟಕ"),
            district=data.get("district", ""),
            district_kn=data.get("district_kn"),
            taluk=data.get("taluk"),
            taluk_kn=data.get("taluk_kn"),
            village=data.get("village"),
            village_kn=data.get("village_kn"),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            lgd_code=data.get("lgd_code"),
            source=data.get("source", "Local Government Directory (LGD), Ministry of Panchayati Raj, Government of India & KSRSAC"),
            source_version=data.get("source_version", "2026-LGD-KRN-v1.0"),
            last_updated=data.get("last_updated", "2026-08"),
        )
