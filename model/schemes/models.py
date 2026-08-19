"""
Data models for the RaithaMitra Karnataka + Central Government Agricultural Scheme Module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


# Allowed status values for scheme verification
VALID_VERIFICATION_STATUSES: Set[str] = {
    "ACTIVE_VERIFIED",
    "ACTIVE_WITH_STATE_IMPLEMENTATION_DETAILS",
    "STATUS_UNCERTAIN",
    "INACTIVE_OR_REPLACED",
    "DO_NOT_USE",
}

# Statuses permitted for standard farmer recommendations
ACTIVE_RECOMMENDED_STATUSES: Set[str] = {
    "ACTIVE_VERIFIED",
    "ACTIVE_WITH_STATE_IMPLEMENTATION_DETAILS",
}


@dataclass(frozen=True)
class GovernmentScheme:
    """
    Represents an officially verified Central or Karnataka state agricultural scheme.
    """

    id: str
    name_en: str
    name_kn: str
    government_level: str
    department: str
    category: str
    description: str
    purpose: str
    eligible_farmer_types: List[str]
    eligible_crops: Optional[List[str]]
    location_scope: str
    benefit_summary: str
    eligibility_summary: str
    application_method: str
    source_authority: str
    source_document_title: str
    source_url: str
    last_verified: str
    verification_status: str
    application_portal: Optional[str] = None
    official_contact: Optional[str] = None
    documents_required: List[str] = field(default_factory=list)
    deadline_or_season: Optional[str] = None
    validity_notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate required fields and verification status."""
        if not self.id or not self.id.strip():
            raise ValueError("Scheme 'id' must be a non-empty string.")
        if not self.name_en or not self.name_en.strip():
            raise ValueError("Scheme 'name_en' must be a non-empty string.")
        if not self.name_kn or not self.name_kn.strip():
            raise ValueError("Scheme 'name_kn' must be a non-empty string.")
        if not self.description or not self.description.strip():
            raise ValueError("Scheme 'description' must be a non-empty string.")
        if not self.source_authority or not self.source_authority.strip():
            raise ValueError("Scheme 'source_authority' must be a non-empty string.")
        if not self.source_url or not self.source_url.strip():
            raise ValueError("Scheme 'source_url' must be a non-empty string.")
        if not self.last_verified or not self.last_verified.strip():
            raise ValueError("Scheme 'last_verified' must be a non-empty string.")
        if self.verification_status not in VALID_VERIFICATION_STATUSES:
            raise ValueError(
                f"Invalid verification_status '{self.verification_status}'. "
                f"Must be one of: {sorted(VALID_VERIFICATION_STATUSES)}"
            )

    @property
    def is_active(self) -> bool:
        """Returns True only if scheme is actively verified for farmer recommendation."""
        return self.verification_status in ACTIVE_RECOMMENDED_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataclass to clean dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GovernmentScheme:
        """Instantiate GovernmentScheme from dictionary with validation."""
        return cls(
            id=data.get("id", ""),
            name_en=data.get("name_en", ""),
            name_kn=data.get("name_kn", ""),
            government_level=data.get("government_level", ""),
            department=data.get("department", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            purpose=data.get("purpose", ""),
            eligible_farmer_types=data.get("eligible_farmer_types", []),
            eligible_crops=data.get("eligible_crops"),
            location_scope=data.get("location_scope", ""),
            benefit_summary=data.get("benefit_summary", ""),
            eligibility_summary=data.get("eligibility_summary", ""),
            application_method=data.get("application_method", ""),
            application_portal=data.get("application_portal"),
            official_contact=data.get("official_contact"),
            documents_required=data.get("documents_required", []),
            deadline_or_season=data.get("deadline_or_season"),
            verification_status=data.get("verification_status", ""),
            source_authority=data.get("source_authority", ""),
            source_document_title=data.get("source_document_title", ""),
            source_url=data.get("source_url", ""),
            last_verified=data.get("last_verified", ""),
            validity_notes=data.get("validity_notes"),
        )
