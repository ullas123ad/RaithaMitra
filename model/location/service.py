"""
Karnataka Location Service providing deterministic administrative hierarchy
and coordinate lookups for RaithaMitra.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from model.location.models import LocationContext, LocationValidationError

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "location" / "karnataka_locations.json"


class LocationNotFoundError(KeyError):
    """Raised when a requested district, taluk, or village is not found."""
    pass


class LocationService:
    """Deterministic, local Karnataka administrative location and coordinate lookup service."""

    def __init__(self, data_path: Optional[Path | str] = None) -> None:
        """Initialize LocationService with local dataset.

        Args:
            data_path: Optional custom path to karnataka_locations.json.
        """
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._raw_data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._districts: List[Dict[str, Any]] = []

        # Lookup caches: normalized_key -> entity
        self._district_lookup: Dict[str, Dict[str, Any]] = {}
        self._load_dataset()

    def _normalize(self, text: Optional[str]) -> str:
        """Normalize input string for deterministic case/whitespace-insensitive matching."""
        if not text:
            return ""
        # Normalize Unicode forms (NFC)
        normalized = unicodedata.normalize("NFC", str(text).strip())
        # Case fold for English, preserve native Kannada scripts
        return " ".join(normalized.lower().split())

    def _load_dataset(self) -> None:
        """Load and index the Karnataka location dataset from local JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Authoritative Karnataka location dataset missing at: {self.data_path}"
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

        self._metadata = self._raw_data.get("metadata", {})
        self._districts = self._raw_data.get("districts", [])

        # Build index maps
        self._district_lookup.clear()
        for dist in self._districts:
            eng_name = dist.get("name", "")
            kn_name = dist.get("name_kn", "")

            norm_eng = self._normalize(eng_name)
            if norm_eng:
                self._district_lookup[norm_eng] = dist

            if kn_name:
                norm_kn = self._normalize(kn_name)
                self._district_lookup[norm_kn] = dist

        logger.info(
            "Loaded Karnataka LocationService: %d districts from %s",
            len(self._districts),
            self.data_path.name,
        )

    @property
    def source_info(self) -> Dict[str, Any]:
        """Return dataset source metadata."""
        return dict(self._metadata)

    def list_districts(self) -> List[Dict[str, Any]]:
        """Return list of all available Karnataka districts."""
        return [
            {
                "name": d.get("name"),
                "name_kn": d.get("name_kn"),
                "lgd_code": d.get("lgd_code"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
                "taluks_count": len(d.get("taluks", [])),
            }
            for d in self._districts
        ]

    def list_taluks(self, district: str) -> List[Dict[str, Any]]:
        """Return list of taluks for a given district.

        Args:
            district: District name in English or Kannada.

        Returns:
            List of taluk dictionaries.

        Raises:
            LocationNotFoundError: If district is not found.
        """
        dist_record = self._find_district(district)
        return [
            {
                "name": t.get("name"),
                "name_kn": t.get("name_kn"),
                "lgd_code": t.get("lgd_code"),
                "latitude": t.get("latitude"),
                "longitude": t.get("longitude"),
                "villages_count": len(t.get("villages", [])),
            }
            for t in dist_record.get("taluks", [])
        ]

    def list_villages(self, district: str, taluk: str) -> List[Dict[str, Any]]:
        """Return list of villages for a given district and taluk.

        Args:
            district: District name in English or Kannada.
            taluk: Taluk name in English or Kannada.

        Returns:
            List of village dictionaries.

        Raises:
            LocationNotFoundError: If district or taluk is not found.
        """
        dist_record = self._find_district(district)
        taluk_record = self._find_taluk(dist_record, taluk)
        return [
            {
                "name": v.get("name"),
                "name_kn": v.get("name_kn"),
                "lgd_code": v.get("lgd_code"),
                "latitude": v.get("latitude"),
                "longitude": v.get("longitude"),
            }
            for v in taluk_record.get("villages", [])
        ]

    def _find_district(self, district_name: str) -> Dict[str, Any]:
        """Find district record by English or Kannada name."""
        norm_name = self._normalize(district_name)
        if not norm_name:
            raise LocationValidationError("District name cannot be empty.")

        if norm_name in self._district_lookup:
            return self._district_lookup[norm_name]

        # Alternative alias checks (e.g. Bangalore vs Bengaluru)
        aliases = {
            "bangalore": "bengaluru urban",
            "bangalore urban": "bengaluru urban",
            "bangalore rural": "bengaluru rural",
            "bengaluru": "bengaluru urban",
            "shimoga": "shivamogga",
            "bijapur": "vijayapura",
            "bellary": "ballari",
            "gulbarga": "kalaburagi",
            "chikmagalur": "chikkamagaluru",
            "chikmagalore": "chikkamagaluru",
            "chikballapur": "chikkaballapura",
            "chikkaballapur": "chikkaballapura",
        }
        if norm_name in aliases and aliases[norm_name] in self._district_lookup:
            return self._district_lookup[aliases[norm_name]]

        raise LocationNotFoundError(
            f"District '{district_name}' not found in Karnataka location database."
        )

    def _find_taluk(self, district_record: Dict[str, Any], taluk_name: str) -> Dict[str, Any]:
        """Find taluk record within a district record."""
        norm_name = self._normalize(taluk_name)
        if not norm_name:
            raise LocationValidationError("Taluk name cannot be empty.")

        for t in district_record.get("taluks", []):
            if self._normalize(t.get("name")) == norm_name or self._normalize(t.get("name_kn")) == norm_name:
                return t

        # Check aliases
        aliases = {
            "bangalore north": "bangalore north",
            "bengaluru north": "bangalore north",
            "bangalore south": "bangalore south",
            "bengaluru south": "bangalore south",
            "bangalore east": "bangalore east",
            "bengaluru east": "bangalore east",
            "hubli": "hubballi",
        }
        if norm_name in aliases:
            target = aliases[norm_name]
            for t in district_record.get("taluks", []):
                if self._normalize(t.get("name")) == target:
                    return t

        dist_name = district_record.get("name", "Unknown")
        raise LocationNotFoundError(
            f"Taluk '{taluk_name}' not found under district '{dist_name}'."
        )

    def _find_village(self, taluk_record: Dict[str, Any], village_name: str, district_name: str) -> Dict[str, Any]:
        """Find village record within a taluk record."""
        norm_name = self._normalize(village_name)
        if not norm_name:
            raise LocationValidationError("Village name cannot be empty.")

        for v in taluk_record.get("villages", []):
            if self._normalize(v.get("name")) == norm_name or self._normalize(v.get("name_kn")) == norm_name:
                return v

        taluk_name = taluk_record.get("name", "Unknown")
        raise LocationNotFoundError(
            f"Village '{village_name}' not found under taluk '{taluk_name}', district '{district_name}'."
        )

    def get_location(
        self,
        district: str,
        taluk: Optional[str] = None,
        village: Optional[str] = None,
    ) -> LocationContext:
        """Resolve a structured location hierarchy and return a validated LocationContext.

        Args:
            district: District name in English or Kannada (e.g. 'Mandya', 'ಮಂಡ್ಯ').
            taluk: Optional Taluk name in English or Kannada (e.g. 'Maddur', 'ಮದ್ದೂರು').
            village: Optional Village name in English or Kannada (e.g. 'Besagarahalli').

        Returns:
            Validated LocationContext with exact coordinates and source metadata.

        Raises:
            LocationNotFoundError: If the requested entity is not found.
            LocationValidationError: If inputs or coordinates are invalid.
        """
        dist_record = self._find_district(district)
        district_name = dist_record.get("name", "")
        district_kn = dist_record.get("name_kn")

        # Case 1: Village specified
        if village:
            if not taluk:
                # If village is specified without taluk, search across all taluks in the district
                matching_village = None
                matching_taluk = None
                for t in dist_record.get("taluks", []):
                    try:
                        v = self._find_village(t, village, district_name)
                        matching_village = v
                        matching_taluk = t
                        break
                    except LocationNotFoundError:
                        continue

                if not matching_village or not matching_taluk:
                    raise LocationNotFoundError(
                        f"Village '{village}' not found in district '{district_name}'."
                    )
                taluk_record = matching_taluk
                village_record = matching_village
            else:
                taluk_record = self._find_taluk(dist_record, taluk)
                village_record = self._find_village(taluk_record, village, district_name)

            return LocationContext(
                state=self._metadata.get("state", "Karnataka"),
                state_kn=self._metadata.get("state_kn", "ಕರ್ನಾಟಕ"),
                district=district_name,
                district_kn=district_kn,
                taluk=taluk_record.get("name"),
                taluk_kn=taluk_record.get("name_kn"),
                village=village_record.get("name"),
                village_kn=village_record.get("name_kn"),
                latitude=float(village_record.get("latitude", 0.0)),
                longitude=float(village_record.get("longitude", 0.0)),
                lgd_code=village_record.get("lgd_code"),
                source=self._metadata.get("source", "Local Government Directory (LGD)"),
                source_version=self._metadata.get("source_version", "2026-LGD-KRN-v1.0"),
                last_updated=self._metadata.get("last_updated", "2026-08"),
            )

        # Case 2: Taluk specified without village
        if taluk:
            taluk_record = self._find_taluk(dist_record, taluk)
            return LocationContext(
                state=self._metadata.get("state", "Karnataka"),
                state_kn=self._metadata.get("state_kn", "ಕರ್ನಾಟಕ"),
                district=district_name,
                district_kn=district_kn,
                taluk=taluk_record.get("name"),
                taluk_kn=taluk_record.get("name_kn"),
                village=None,
                village_kn=None,
                latitude=float(taluk_record.get("latitude", 0.0)),
                longitude=float(taluk_record.get("longitude", 0.0)),
                lgd_code=taluk_record.get("lgd_code"),
                source=self._metadata.get("source", "Local Government Directory (LGD)"),
                source_version=self._metadata.get("source_version", "2026-LGD-KRN-v1.0"),
                last_updated=self._metadata.get("last_updated", "2026-08"),
            )

        # Case 3: District only
        return LocationContext(
            state=self._metadata.get("state", "Karnataka"),
            state_kn=self._metadata.get("state_kn", "ಕರ್ನಾಟಕ"),
            district=district_name,
            district_kn=district_kn,
            taluk=None,
            taluk_kn=None,
            village=None,
            village_kn=None,
            latitude=float(dist_record.get("latitude", 0.0)),
            longitude=float(dist_record.get("longitude", 0.0)),
            lgd_code=dist_record.get("lgd_code"),
            source=self._metadata.get("source", "Local Government Directory (LGD)"),
            source_version=self._metadata.get("source_version", "2026-LGD-KRN-v1.0"),
            last_updated=self._metadata.get("last_updated", "2026-08"),
        )

    def search_location(self, query: str, limit: int = 10) -> List[LocationContext]:
        """Search across districts, taluks, and villages using a free-form string.

        Args:
            query: Search query in English or Kannada.
            limit: Maximum number of results to return.

        Returns:
            List of matching LocationContext objects.
        """
        norm_q = self._normalize(query)
        if not norm_q:
            return []

        results: List[LocationContext] = []

        # 1. Check districts
        for dist in self._districts:
            d_name = dist.get("name", "")
            d_kn = dist.get("name_kn", "")
            if norm_q in self._normalize(d_name) or (d_kn and norm_q in self._normalize(d_kn)):
                results.append(self.get_location(district=d_name))
                if len(results) >= limit:
                    return results

        # 2. Check taluks
        for dist in self._districts:
            d_name = dist.get("name", "")
            for taluk in dist.get("taluks", []):
                t_name = taluk.get("name", "")
                t_kn = taluk.get("name_kn", "")
                if norm_q in self._normalize(t_name) or (t_kn and norm_q in self._normalize(t_kn)):
                    results.append(self.get_location(district=d_name, taluk=t_name))
                    if len(results) >= limit:
                        return results

        # 3. Check villages
        for dist in self._districts:
            d_name = dist.get("name", "")
            for taluk in dist.get("taluks", []):
                t_name = taluk.get("name", "")
                for village in taluk.get("villages", []):
                    v_name = village.get("name", "")
                    v_kn = village.get("name_kn", "")
                    if norm_q in self._normalize(v_name) or (v_kn and norm_q in self._normalize(v_kn)):
                        results.append(self.get_location(district=d_name, taluk=t_name, village=v_name))
                        if len(results) >= limit:
                            return results

        return results

    def detect_location_from_text(self, text: str) -> Optional[LocationContext]:
        """Detect explicit Karnataka district mention in spoken or written farmer text.

        Handles English and Kannada district names with native Kannada locative suffixes
        (e.g., 'ರಲ್ಲಿ', 'ದಲ್ಲಿ', 'ಯಲ್ಲಿ', 'ನಲ್ಲಿ', 'ಅಲ್ಲಿ').
        """
        import re

        if not text or not text.strip():
            return None

        clean_text = self._normalize(text)

        # Check all known districts & aliases
        for dist in self._districts:
            eng_name = dist.get("name", "").lower()
            kn_name = dist.get("name_kn", "")

            # Check English name (word boundary check)
            if eng_name and re.search(r"\b" + re.escape(eng_name) + r"\b", clean_text, re.IGNORECASE):
                return self.get_location(district=dist["name"])

            # Check Kannada name stem & common locative suffixes
            if kn_name:
                stem = kn_name.strip()
                patterns = [
                    re.escape(stem) + r"(ದಲ್ಲಿ|ಯಲ್ಲಿ|ನಲ್ಲಿ|ರಲ್ಲಿ|ಅಲ್ಲಿ|ಗೆ|ಯ|ನ|ದಿಂದ)?",
                ]
                if stem in text or any(re.search(p, text) for p in patterns):
                    return self.get_location(district=dist["name"])

        # Alias mapping for English and Kannada district names & stems
        alias_map = {
            "ಬೆಂಗಳೂರು": "Bengaluru Urban", "bengaluru": "Bengaluru Urban", "bangalore": "Bengaluru Urban", "ಬೆಂಗಳೂ": "Bengaluru Urban",
            "ಮೈಸೂರು": "Mysuru", "mysore": "Mysuru", "mysuru": "Mysuru", "ಮೈಸೂ": "Mysuru",
            "ಶಿವಮೊಗ್ಗ": "Shivamogga", "shimoga": "Shivamogga", "shivamogga": "Shivamogga", "ಶಿವಮೊ": "Shivamogga",
            "ಬಳ್ಳಾರಿ": "Ballari", "bellary": "Ballari", "ballari": "Ballari", "ಬಳ್ಳಾ": "Ballari",
            "ಕಲಬುರಗಿ": "Kalaburagi", "gulbarga": "Kalaburagi", "kalaburagi": "Kalaburagi", "ಕಲಬು": "Kalaburagi",
            "ವಿಜಯಪುರ": "Vijayapura", "bijapur": "Vijayapura", "vijayapura": "Vijayapura", "ವಿಜಯ": "Vijayapura",
            "ಚಿಕ್ಕಮಗಳೂರು": "Chikkamagaluru", "chikmagalur": "Chikkamagaluru", "chikkamagaluru": "Chikkamagaluru", "ಚಿಕ್ಕಮ": "Chikkamagaluru",
            "ಉಡುಪಿ": "Udupi", "udupi": "Udupi", "ಉಡುಪ": "Udupi",
            "ಬೆಳಗಾವಿ": "Belagavi", "belgaum": "Belagavi", "belagavi": "Belagavi", "ಬೆಳಗಾ": "Belagavi",
            "ಮಂಡ್ಯ": "Mandya", "mandya": "Mandya",
            "ಹಾವೇರಿ": "Haveri", "haveri": "Haveri", "ಹಾವೇ": "Haveri",
            "ಕೋಲಾರ": "Kolar", "kolar": "Kolar", "ಕೋಲಾ": "Kolar",
            "ಹಾಸನ": "Hassan", "hassan": "Hassan", "ಹಾಸ": "Hassan",
            "ತುಮಕೂರು": "Tumakuru", "tumkur": "Tumakuru", "tumakuru": "Tumakuru", "ತುಮಕೂ": "Tumakuru",
            "ದಾವಣಗೆರೆ": "Davanagere", "davangere": "Davanagere", "davanagere": "Davanagere", "ದಾವಣ": "Davanagere",
            "ಧಾರವಾಡ": "Dharwad", "dharwad": "Dharwad", "ಧಾರ್ವಾ": "Dharwad"
        }

        for alias, canon_dist in alias_map.items():
            if alias in text or re.search(r"\b" + re.escape(alias) + r"\b", clean_text, re.IGNORECASE):
                try:
                    return self.get_location(district=canon_dist)
                except LocationNotFoundError:
                    pass

        return None

    def get_location_from_coordinates(self, latitude: float, longitude: float) -> LocationContext:
        """Resolve GPS latitude and longitude coordinates to nearest supported Karnataka district.

        Raises:
            LocationValidationError: If coordinates are outside Karnataka bounding box.
        """
        import math

        lat, lon = float(latitude), float(longitude)

        # Karnataka bounding box validation (~11.0°N to 19.0°N, 73.5°E to 79.0°E)
        if not (11.0 <= lat <= 19.0 and 73.5 <= lon <= 79.0):
            raise LocationValidationError(
                "RaithaMitra currently provides Karnataka-focused agricultural advisory. "
                f"Coordinates ({lat:.4f}, {lon:.4f}) are outside Karnataka state bounds."
            )

        min_dist = float("inf")
        nearest_dist_record = None

        for dist in self._districts:
            d_lat = float(dist.get("latitude", 0.0))
            d_lon = float(dist.get("longitude", 0.0))
            if d_lat and d_lon:
                dist_val = math.sqrt((lat - d_lat) ** 2 + (lon - d_lon) ** 2)
                if dist_val < min_dist:
                    min_dist = dist_val
                    nearest_dist_record = dist

        if not nearest_dist_record:
            raise LocationValidationError("Could not resolve GPS coordinates to a Karnataka district.")

        return self.get_location(district=nearest_dist_record["name"])
