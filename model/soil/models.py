"""
Data models representing Karnataka Soil Health Context for RaithaMitra.

Maintains strict separation between regional soil classifications and
actual field-specific laboratory soil measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SoilContext:
    """
    Structured soil context for agricultural advisory and nutrient management.

    Attributes:
        available: Boolean indicating whether valid soil information was retrieved.
        is_measured_data: True ONLY if actual laboratory soil measurements (pH, N, P, K)
                         are provided by the farmer/testing lab. False for regional profiles.
        district: District name in Karnataka.
        taluk: Optional Taluk name.
        village: Optional Village name.
        agro_climatic_zone: Karnataka Agro-Climatic Zone (e.g., Southern Dry Zone).
        dominant_soil_types: List of primary soil types in this region (e.g. Red sandy loam).
        soil_order: Soil taxonomy order (e.g. Alfisols, Vertisols, Ultisols).
        texture: Physical soil texture description.
        drainage: Soil drainage and aeration characteristics.
        typical_ph_range: Expected regional pH range (e.g. 6.2 - 7.5). NOT a field test.
        ph: Actual laboratory-measured soil pH (None if unmeasured).
        nitrogen: Actual laboratory-measured available nitrogen (kg/ha) (None if unmeasured).
        phosphorus: Actual laboratory-measured available phosphorus (kg/ha) (None if unmeasured).
        potassium: Actual laboratory-measured available potassium (kg/ha) (None if unmeasured).
        organic_carbon: Actual laboratory-measured organic carbon (%) (None if unmeasured).
        electrical_conductivity: Actual measured EC (dS/m) (None if unmeasured).
        micronutrients: Actual measured or regional micronutrient status dictionary.
        general_fertility_status: General fertility status dictionary for the agro-climatic region.
        management_recommendations: Verified soil management practices.
        suitable_crops: Crops well-suited to this soil regime.
        source_authority: Authoritative agency/institution providing the soil data.
        source_document: Official publication or portal reference.
        retrieved_at: ISO-8601 UTC timestamp of retrieval.
        status_message: Descriptive status message or error note.
    """

    available: bool = True
    is_measured_data: bool = False
    district: Optional[str] = None
    taluk: Optional[str] = None
    village: Optional[str] = None
    agro_climatic_zone: Optional[str] = None
    dominant_soil_types: List[str] = field(default_factory=list)
    soil_order: Optional[str] = None
    texture: Optional[str] = None
    drainage: Optional[str] = None
    typical_ph_range: Optional[str] = None
    ph: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    organic_carbon: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    micronutrients: Optional[Dict[str, Any]] = None
    general_fertility_status: Optional[Dict[str, Any]] = None
    management_recommendations: Optional[str] = None
    suitable_crops: List[str] = field(default_factory=list)
    source_authority: str = ""
    source_document: str = ""
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status_message: str = "Success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SoilContext:
        """Create SoilContext instance from dictionary representation."""
        return cls(
            available=data.get("available", True),
            is_measured_data=data.get("is_measured_data", False),
            district=data.get("district"),
            taluk=data.get("taluk"),
            village=data.get("village"),
            agro_climatic_zone=data.get("agro_climatic_zone"),
            dominant_soil_types=data.get("dominant_soil_types", []),
            soil_order=data.get("soil_order"),
            texture=data.get("texture"),
            drainage=data.get("drainage"),
            typical_ph_range=data.get("typical_ph_range"),
            ph=data.get("ph"),
            nitrogen=data.get("nitrogen"),
            phosphorus=data.get("phosphorus"),
            potassium=data.get("potassium"),
            organic_carbon=data.get("organic_carbon"),
            electrical_conductivity=data.get("electrical_conductivity"),
            micronutrients=data.get("micronutrients"),
            general_fertility_status=data.get("general_fertility_status"),
            management_recommendations=data.get("management_recommendations"),
            suitable_crops=data.get("suitable_crops", []),
            source_authority=data.get("source_authority", ""),
            source_document=data.get("source_document", ""),
            retrieved_at=data.get(
                "retrieved_at", datetime.now(timezone.utc).isoformat()
            ),
            status_message=data.get("status_message", "Success"),
        )
