"""
Karnataka Soil Health Context Service for RaithaMitra.

Provides lightweight, CPU-efficient, deterministic soil context based on
authoritative ICAR / KSDA agro-climatic regional classifications and optional
farmer-provided laboratory measurements.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from model.location.models import LocationContext
from model.advisory.crop_identifier import normalize_crop_name
from model.soil.models import SoilContext

logger = logging.getLogger(__name__)


class SoilServiceError(Exception):
    """Raised when soil profile loading or processing fails."""
    pass


class SoilService:
    """
    Retrieves authoritative regional soil profiles and integrates field-specific
    soil test data for Karnataka agricultural locations.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize SoilService and load regional Karnataka soil profiles."""
        self.data_path = data_path or self._get_default_data_path()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._load_profiles()

    def _get_default_data_path(self) -> str:
        """Resolve path to default karnataka_soil_profiles.json."""
        project_root = Path(__file__).resolve().parent.parent.parent
        return str(project_root / "data" / "soil" / "karnataka_soil_profiles.json")

    def _load_profiles(self) -> None:
        """Load and index soil profiles from JSON."""
        if not os.path.exists(self.data_path):
            raise SoilServiceError(f"Soil profiles dataset not found at: {self.data_path}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise SoilServiceError("Soil profiles dataset format must be a list of JSON objects.")

            loaded: Dict[str, Dict[str, Any]] = {}
            for item in data:
                dist = item.get("district", "").strip().lower()
                if dist:
                    loaded[dist] = item

            self._profiles = loaded
            logger.info("Loaded %d Karnataka regional soil profiles", len(self._profiles))

        except Exception as e:
            raise SoilServiceError(f"Failed to load soil profiles: {e}") from e

    @property
    def total_profiles_count(self) -> int:
        """Total number of loaded regional soil profiles."""
        return len(self._profiles)

    def get_soil_context(
        self,
        location: Optional[LocationContext] = None,
        district: Optional[str] = None,
        taluk: Optional[str] = None,
        village: Optional[str] = None,
        crop: Optional[str] = None,
        measured_data: Optional[Dict[str, Any]] = None,
    ) -> SoilContext:
        """
        Retrieves structured soil context for a given Karnataka location.

        Args:
            location: Optional LocationContext object.
            district: Optional district name string.
            taluk: Optional taluk name string.
            village: Optional village name string.
            crop: Optional crop name string.
            measured_data: Optional dictionary containing actual laboratory soil test
                           measurements (e.g. ph, nitrogen, phosphorus, potassium, organic_carbon).

        Returns:
            SoilContext dataclass instance.
        """
        # Resolve target district name
        target_district = ""
        if location and location.district:
            target_district = location.district.strip()
            taluk = taluk or location.taluk
            village = village or location.village
        elif district and district.strip():
            target_district = district.strip()

        if not target_district:
            return SoilContext(
                available=False,
                status_message="Location or district information is required to resolve soil context."
            )

        dist_key = target_district.lower()
        profile = self._profiles.get(dist_key)

        if not profile:
            return SoilContext(
                available=False,
                district=target_district,
                taluk=taluk,
                village=village,
                status_message=f"No authoritative soil profile found for district '{target_district}'."
            )

        canonical_crop = normalize_crop_name(crop) if crop else None

        # Build base context from verified regional profile
        context = SoilContext(
            available=True,
            is_measured_data=False,
            district=profile.get("district", target_district),
            taluk=taluk,
            village=village,
            agro_climatic_zone=profile.get("agro_climatic_zone"),
            dominant_soil_types=profile.get("dominant_soil_types", []),
            soil_order=profile.get("soil_order"),
            texture=profile.get("texture"),
            drainage=profile.get("drainage"),
            typical_ph_range=profile.get("typical_ph_range"),
            ph=None,
            nitrogen=None,
            phosphorus=None,
            potassium=None,
            organic_carbon=None,
            electrical_conductivity=None,
            micronutrients=None,
            general_fertility_status=profile.get("general_fertility_status"),
            management_recommendations=profile.get("management_recommendations"),
            suitable_crops=profile.get("major_crops_suited", []),
            source_authority=profile.get("source_authority", "ICAR-NBSS&LUP & KSDA"),
            source_document=profile.get("source_document", "Soils of Karnataka, ICAR-NBSS&LUP Publ. 47"),
            status_message="Success"
        )

        # Incorporate actual laboratory measurements if explicitly provided
        if measured_data and isinstance(measured_data, dict):
            context.is_measured_data = True
            context.ph = measured_data.get("ph")
            context.nitrogen = measured_data.get("nitrogen")
            context.phosphorus = measured_data.get("phosphorus")
            context.potassium = measured_data.get("potassium")
            context.organic_carbon = measured_data.get("organic_carbon")
            context.electrical_conductivity = measured_data.get("electrical_conductivity")
            context.micronutrients = measured_data.get("micronutrients")

        return context

    def format_soil_context(self, context: SoilContext) -> str:
        """
        Formats SoilContext into a clean, factual, structured block for LLM prompt injection.
        """
        if not context or not context.available:
            return ""

        lines: List[str] = []
        if context.is_measured_data:
            lines.append("--- FIELD-MEASURED SOIL TEST RECORD (LABORATORY VERIFIED) ---")
        else:
            lines.append("--- REGIONAL SOIL HEALTH PROFILE (GENERAL CLASSIFICATION) ---")

        loc_label = context.district or "Karnataka"
        if context.taluk:
            loc_label = f"{context.taluk}, {loc_label}"
        if context.village:
            loc_label = f"{context.village}, {loc_label}"
        lines.append(f"Location: {loc_label}")

        if context.agro_climatic_zone:
            lines.append(f"Agro-Climatic Zone: {context.agro_climatic_zone}")

        if context.dominant_soil_types:
            lines.append(f"Dominant Soil Types: {', '.join(context.dominant_soil_types)}")

        if context.soil_order:
            lines.append(f"Soil Taxonomy Order: {context.soil_order}")

        if context.texture:
            lines.append(f"Soil Texture: {context.texture}")

        if context.drainage:
            lines.append(f"Drainage & Aeration: {context.drainage}")

        if context.typical_ph_range:
            lines.append(f"Typical Regional pH Range: {context.typical_ph_range}")

        # Measured laboratory values (ONLY displayed if actually provided)
        if context.is_measured_data:
            meas_parts = []
            if context.ph is not None:
                meas_parts.append(f"pH: {context.ph}")
            if context.organic_carbon is not None:
                meas_parts.append(f"Organic Carbon: {context.organic_carbon}%")
            if context.nitrogen is not None:
                meas_parts.append(f"Available N: {context.nitrogen} kg/ha")
            if context.phosphorus is not None:
                meas_parts.append(f"Available P: {context.phosphorus} kg/ha")
            if context.potassium is not None:
                meas_parts.append(f"Available K: {context.potassium} kg/ha")
            if context.electrical_conductivity is not None:
                meas_parts.append(f"EC: {context.electrical_conductivity} dS/m")
            if meas_parts:
                lines.append(f"Measured Soil Nutrients: {', '.join(meas_parts)}")

        # General fertility status from regional survey
        if context.general_fertility_status:
            gfs = context.general_fertility_status
            fert_parts = []
            if "organic_carbon_status" in gfs:
                fert_parts.append(f"Organic Carbon: {gfs['organic_carbon_status']}")
            if "available_nitrogen_status" in gfs:
                fert_parts.append(f"Nitrogen: {gfs['available_nitrogen_status']}")
            if "available_phosphorus_status" in gfs:
                fert_parts.append(f"Phosphorus: {gfs['available_phosphorus_status']}")
            if "available_potassium_status" in gfs:
                fert_parts.append(f"Potassium: {gfs['available_potassium_status']}")
            if "major_micronutrient_deficiencies" in gfs:
                defs = ", ".join(gfs["major_micronutrient_deficiencies"])
                fert_parts.append(f"Common Deficiencies: {defs}")
            if fert_parts:
                lines.append(f"Regional Fertility Status: {'; '.join(fert_parts)}")

        if context.management_recommendations:
            lines.append(f"Soil Management Guidance: {context.management_recommendations}")

        lines.append(f"Source Authority: {context.source_authority} ({context.source_document})")

        lines.append("")
        lines.append("Important Soil Guidance Rules:")
        if not context.is_measured_data:
            lines.append("1. This is regional soil classification data, NOT the farmer's specific field test.")
            lines.append("2. Do NOT state that the farmer's soil has specific measured N/P/K values or exact pH.")
            lines.append("3. Do NOT prescribe exact chemical fertilizer dosages without an official Soil Health Card test.")
            lines.append("4. Advise the farmer to test their soil through the Soil Health Card scheme at their local Raitha Samparka Kendra (RSK).")
        else:
            lines.append("1. Measured soil test values are available. Tailor nutrient guidance to the measured pH and N-P-K levels.")
            lines.append("2. Follow standard University of Agricultural Sciences (UAS) package of practices for fertilizer recommendations.")

        return "\n".join(lines).strip()
