"""
Karnataka + Central Government Agricultural Scheme Service for RaithaMitra.

Provides lightweight, deterministic, CPU-efficient search and retrieval of
authoritative, officially verified agricultural schemes for Karnataka farmers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from model.location.models import LocationContext
from model.advisory.crop_identifier import normalize_crop_name, detect_crop_from_text
from model.schemes.models import GovernmentScheme

logger = logging.getLogger(__name__)


class SchemeServiceError(Exception):
    """Raised when scheme loading or retrieval fails."""
    pass


class SchemeService:
    """
    Search and retrieval service for verified Central and Karnataka State
    Government agricultural schemes.
    """

    # Canonical aliases mapping English and Kannada queries to specific scheme IDs
    SCHEME_ALIASES: Dict[str, List[str]] = {
        "pm_kisan": [
            "pm-kisan", "pm kisan", "pmkisan", "kisan samman nidhi", "kisan samman",
            "ಪಿಎಂ ಕಿಸಾನ್", "ಪಿಎಂ-ಕಿಸಾನ್", "ಕಿಸಾನ್ ಸಮ್ಮಾನ್", "ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ",
            "ಪ್ರಧಾನ ಮಂತ್ರಿ ಕಿಸಾನ್", "ಪ್ರಧಾನಮಂತ್ರಿ ಕಿಸಾನ್", "6000", "financial support",
            "ಆದಾಯ ಬೆಂಬಲ", "ರೈತರ ಖಾತೆಗೆ ಹಣ"
        ],
        "pmfby_karnataka": [
            "pmfby", "pm fby", "fasal bima", "fasal bima yojana", "crop insurance",
            "crop loss", "crop damage", "rain loss", "drought compensation",
            "ಬೆಳೆ ವಿಮೆ", "ಫಸಲ್ ಬಿಮಾ", "ಫಸಲ್ ಭೀಮಾ", "ವಿಮೆ", "ಬೆಳೆ ಹಾನಿ", "ಬೆಳೆ ನಷ್ಟ",
            "ಬೆಳೆ ಪರಿಹಾರ", "ಮಳೆ ಹಾನಿ", "samrakshane", "ಸಂರಕ್ಷಣೆ"
        ],
        "kcc_credit_support": [
            "kcc", "kisan credit card", "crop loan", "farm loan", "agricultural loan",
            "interest subvention", "4% loan", "ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್", "ಕಿಸಾನ್ ಕಾರ್ಡ್",
            "ಬೆಳೆ ಸಾಲ", "ಕೃಷಿ ಸಾಲ", "ಬಡ್ಡಿ ರಿಯಾಯಿತಿ", "ಬಡ್ಡಿ ಸಹಾಯಧನ", "ಬ್ಯಾಂಕ್ ಸಾಲ"
        ],
        "soil_health_card": [
            "soil health", "soil health card", "shc", "soil test", "soil testing",
            "fertilizer recommendation", "ಮಣ್ಣು ಆರೋಗ್ಯ", "ಮಣ್ಣು ಆರೋಗ್ಯ ಪತ್ರಿಕೆ",
            "ಮಣ್ಣು ಪರೀಕ್ಷೆ", "ರಸಗೊಬ್ಬರ ಪ್ರಮಾಣ", "ಮಣ್ಣಿನ ಫಲವತ್ತತೆ"
        ],
        "karnataka_krishi_bhagya": [
            "krishi bhagya", "krishibhagya", "farm pond", "krishi honda", "polythene lining",
            "diesel pump", "rainfed scheme", "ಕೃಷಿ ಭಾಗ್ಯ", "ಕೃಷಿಭಾಗ್ಯ", "ಕೃಷಿ ಹೊಂಡ",
            "ಕೃಷಿಹೊಂಡ", "ಹೊಂಡ", "ಪಾಲಿಥಿನ್ ಹೊದಿಕೆ", "ಮಳೆನೀರು ಕೊಯ್ಲು"
        ],
        "karnataka_raita_siri": [
            "raita siri", "raitha siri", "raitasiri", "millet scheme", "millet incentive",
            "ragi incentive", "siridhanya", "ರೈತ ಸಿರಿ", "ರೈತಸಿರಿ", "ಸಿರಿಧಾನ್ಯ ಪ್ರೋತ್ಸಾಹಧನ",
            "ಸಿರಿಧಾನ್ಯ", "ರಾಗಿ ಪ್ರೋತ್ಸಾಹಧನ", "ಸಿರಿ ಧಾನ್ಯ ಯೋಜನೆ"
        ],
        "pmksy_per_drop_more_crop": [
            "pmksy", "per drop more crop", "drip irrigation", "sprinkler irrigation",
            "micro irrigation", "irrigation subsidy", "ಹನಿ ನೀರಾವರಿ", "ತುಂತುರು ನೀರಾವರಿ",
            "ಸೂಕ್ಷ್ಮ ನೀರಾವರಿ", "ಕೃಷಿ ಸಿಂಚಾಯಿ", "ನೀರಾವರಿ ಸಹಾಯಧನ"
        ],
        "smam_mechanization_karnataka": [
            "smam", "mechanization", "farm machinery", "tractor subsidy", "tiller",
            "farm equipment", "custom hiring centre", "ಕೃಷಿ ಯಾಂತ್ರೀಕರಣ", "ಯಂತ್ರೋಪಕರಣ",
            "ಟ್ರಾಕ್ಟರ್ ಸಹಾಯಧನ", "ಪವರ್ ಟಿಲ್ಲರ್", "ಕೃಷಿ ಉಪಕರಣ"
        ],
        "midh_horticulture_karnataka": [
            "midh", "nhm", "horticulture scheme", "polyhouse", "shade net", "orchard subsidy",
            "cold storage", "ತೋಟಗಾರಿಕೆ", "ತೋಟಗಾರಿಕಾ ಯೋಜನೆ", "ಪಾಲಿಹೌಸ್", "ಶೇಡ್ ನೆಟ್",
            "ಹಣ್ಣು ತರಕಾರಿ ಯೋಜನೆ"
        ],
        "karnataka_fruits_portal": [
            "fruits", "fruits id", "fid", "fruits portal", "farmer registration",
            "ಫ್ರೂಟ್ಸ್", "ಫ್ರೂಟ್ಸ್ ಪೋರ್ಟಲ್", "ಫ್ರೂಟ್ಸ್ ಐಡಿ", "ಎಫ್‌ಐಡಿ", "ರೈತರ ನೋಂದಣಿ"
        ],
    }

    # Generic agriculture scheme intent keywords
    GENERAL_SCHEME_KEYWORDS: Set[str] = {
        "scheme", "schemes", "subsidy", "subsidies", "yojana", "yojane", "benefit", "benefits",
        "government", "assistance", "support", "relief", "grant", "funds",
        "ಯೋಜನೆ", "ಯೋಜನೆಗಳು", "ಸರ್ಕಾರಿ ಯೋಜನೆ", "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು", "ಸಹಾಯಧನ", "ಪ್ರೋತ್ಸಾಹಧನ",
        "ಪರಿಹಾರ", "ಸರ್ಕಾರದ ಯೋಜನೆಗಳು", "ಸರ್ಕಾರದಿಂದ", "ರೈತರಿಗೆ ಯೋಜನೆ", "ಅನುದಾನ", "ಸೌಲಭ್ಯ"
    }

    # Agricultural domain validation terms
    AGRI_DOMAIN_KEYWORDS: Set[str] = {
        "farmer", "farmers", "farm", "farming", "crop", "crops", "agriculture", "agricultural",
        "cultivation", "seed", "seeds", "soil", "water", "irrigation", "rain", "drought",
        "paddy", "ragi", "maize", "groundnut", "cotton", "sugarcane", "chilli", "onion",
        "potato", "banana", "tomato", "millet", "horticulture",
        "ರೈತ", "ರೈತರಿಗೆ", "ಕೃಷಿ", "ಬೆಳೆ", "ಬೆಳೆಗಳು", "ಜಮೀನು", "ಭೂಮಿ", "ಮಳೆ", "ನೀರು",
        "ರಾಗಿ", "ಭತ್ತ", "ಮೆಕ್ಕೆಜೋಳ", "ಕಡಲೆಕಾಯಿ", "ಕಬ್ಬು", "ಹತ್ತಿ", "ಮೆಣಸಿನಕಾಯಿ",
        "ಈರುಳ್ಳಿ", "ಆಲೂಗಡ್ಡೆ", "ಬಾಳೆ", "ಟೊಮ್ಯಾಟೊ", "ತೋಟಗಾರಿಕೆ", "ಸಿರಿಧಾನ್ಯ"
    }

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize SchemeService and load verified scheme dataset."""
        self.data_path = data_path or self._get_default_data_path()
        self._schemes: Dict[str, GovernmentScheme] = {}
        self._load_dataset()

    def _get_default_data_path(self) -> str:
        """Resolve path to default government_schemes.json."""
        project_root = Path(__file__).resolve().parent.parent.parent
        return str(project_root / "data" / "schemes" / "government_schemes.json")

    def _load_dataset(self) -> None:
        """Load and strictly validate scheme records from JSON."""
        if not os.path.exists(self.data_path):
            raise SchemeServiceError(f"Government schemes dataset not found at: {self.data_path}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise SchemeServiceError("Schemes dataset format must be a list of JSON objects.")

            loaded: Dict[str, GovernmentScheme] = {}
            for item in data:
                scheme = GovernmentScheme.from_dict(item)
                if scheme.id in loaded:
                    raise SchemeServiceError(f"Duplicate scheme ID found: {scheme.id}")
                loaded[scheme.id] = scheme

            self._schemes = loaded
            logger.info("Loaded %d government schemes (%d active)", len(self._schemes), len(self.list_active_schemes()))

        except Exception as e:
            raise SchemeServiceError(f"Failed to load government schemes: {e}") from e

    @property
    def total_count(self) -> int:
        """Total number of loaded scheme records."""
        return len(self._schemes)

    def get_scheme(self, scheme_id: str) -> Optional[GovernmentScheme]:
        """Fetch a single scheme record by exact ID."""
        return self._schemes.get(scheme_id)

    def list_active_schemes(self) -> List[GovernmentScheme]:
        """List all officially active and verified schemes."""
        return [s for s in self._schemes.values() if s.is_active]

    def _is_agricultural_query(self, query: str) -> bool:
        """Checks if the query is in the agricultural or scheme domain."""
        q_lower = query.lower()
        # Direct alias check
        for aliases in self.SCHEME_ALIASES.values():
            if any(alias in q_lower for alias in aliases):
                return True
        # Keyword checks
        has_scheme_kw = any(kw in q_lower for kw in self.GENERAL_SCHEME_KEYWORDS)
        has_agri_kw = any(kw in q_lower for kw in self.AGRI_DOMAIN_KEYWORDS)
        return has_scheme_kw or has_agri_kw

    def search_schemes(
        self,
        query: str,
        crop: Optional[str] = None,
        district: Optional[str] = None,
        limit: int = 5,
    ) -> List[GovernmentScheme]:
        """
        Search verified schemes by query text, crop context, and district.
        Only returns active verified schemes.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        if not self._is_agricultural_query(clean_query):
            return []

        q_lower = clean_query.lower()
        active_schemes = self.list_active_schemes()

        # Step 1: Detect exact scheme alias matches
        matched_by_alias: List[GovernmentScheme] = []
        for scheme_id, aliases in self.SCHEME_ALIASES.items():
            scheme = self._schemes.get(scheme_id)
            if scheme and scheme.is_active:
                for alias in aliases:
                    if alias in q_lower:
                        matched_by_alias.append(scheme)
                        break

        # Step 2: Resolve canonical crop
        resolved_crop = normalize_crop_name(crop) if crop else detect_crop_from_text(clean_query)

        # Step 3: Determine if this is a broad general scheme inquiry
        general_patterns = [
            "ಯಾವ ಯೋಜನೆ", "ಯಾವ ಸರ್ಕಾರಿ", "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು", "ಸರ್ಕಾರದ ಯೋಜನೆಗಳು",
            "ರೈತರಿಗೆ ಯೋಜನೆ", "ರೈತರಿಗೆ ಸರ್ಕಾರಿ", "ಕೃಷಿ ಯೋಜನೆಗಳು", "ಯಾವ ಯಾವ ಯೋಜನೆ",
            "ಯೋಜನೆಗಳು ಯಾವುವು", "ಯೋಜನೆಗಳಿವೆಯೇ", "ಯೋಜನೆಗಳು ಇವೆಯೇ", "ಸಹಾಯಧನ ಯೋಜನೆ",
            "ಯೋಜನೆಗಳ ಮಾಹಿತಿ", "ಸಹಾಯಧನ ಇದೆಯೇ",
            "what schemes", "which schemes", "government schemes", "farmer schemes",
            "subsidies available", "list of schemes", "all schemes", "schemes for farmers"
        ]
        is_general_inquiry = any(p in q_lower for p in general_patterns) or (
            ("scheme" in q_lower or "ಯೋಜನೆ" in q_lower or "ಸಹಾಯಧನ" in q_lower) and
            ("what" in q_lower or "which" in q_lower or "ಯಾವ" in q_lower or "ಯಾವುವು" in q_lower or "ಇವೆಯೇ" in q_lower)
        )

        # Step 4: Score all active schemes
        scored_schemes = []
        for scheme in active_schemes:
            score = 0.0

            # Alias match bonus (Highest priority)
            if scheme in matched_by_alias:
                score += 10.0

            # Name match
            if scheme.name_en.lower() in q_lower or scheme.name_kn in clean_query:
                score += 8.0

            # Category / Topic match
            if scheme.category.lower() in q_lower:
                score += 4.0

            # Crop-specific relevance
            if resolved_crop:
                if scheme.eligible_crops:
                    if resolved_crop in scheme.eligible_crops:
                        score += 5.0
                    else:
                        score -= 2.0
                else:
                    # General crop scheme
                    if is_general_inquiry:
                        score += 2.0

            # General scheme inquiry baseline score (only when broad question is asked)
            if is_general_inquiry:
                if scheme.id in {"pm_kisan", "pmfby_karnataka", "karnataka_krishi_bhagya", "kcc_credit_support"}:
                    score += 4.0
                else:
                    score += 2.0

            # Minimum relevance threshold: require at least 3.0 points to return a scheme
            if score >= 3.0:
                scored_schemes.append((score, scheme))

        # Sort descending by score, tie-break by ID
        scored_schemes.sort(key=lambda item: (item[0], item[1].id), reverse=True)

        return [item[1] for item in scored_schemes[:limit]]

    def find_relevant_schemes(
        self,
        query: str,
        crop: Optional[str] = None,
        location: Optional[LocationContext] = None,
        limit: int = 5,
    ) -> List[GovernmentScheme]:
        """
        Find relevant verified schemes using query, resolved crop, and LocationContext.
        """
        district = location.district if location else None
        return self.search_schemes(
            query=query,
            crop=crop,
            district=district,
            limit=limit,
        )

    def format_scheme_context(self, schemes: List[GovernmentScheme]) -> str:
        """
        Format retrieved schemes into a clean, factual, structured block for LLM prompt injection.
        """
        if not schemes:
            return ""

        lines: List[str] = ["--- RELEVANT GOVERNMENT SCHEMES (OFFICIAL VERIFIED) ---"]
        for idx, s in enumerate(schemes, start=1):
            lines.append(f"[{idx}] Scheme: {s.name_en}")
            lines.append(f"    Kannada Name: {s.name_kn}")
            lines.append(f"    Government Level: {s.government_level} | Department: {s.department}")
            lines.append(f"    Category: {s.category}")
            lines.append(f"    Purpose: {s.purpose}")
            lines.append(f"    Benefit: {s.benefit_summary}")
            lines.append(f"    Eligibility Criteria: {s.eligibility_summary}")
            lines.append(f"    Application Method: {s.application_method}")
            if s.application_portal:
                lines.append(f"    Official Portal: {s.application_portal}")
            if s.official_contact:
                lines.append(f"    Contact/Helpline: {s.official_contact}")
            lines.append(f"    Source: {s.source_authority} ({s.source_url})")
            lines.append(f"    Status: {s.verification_status} (Last Verified: {s.last_verified})")
            if s.validity_notes:
                lines.append(f"    Notes: {s.validity_notes}")
            lines.append("")

        lines.append("Important Rules for Scheme Guidance:")
        lines.append("1. Present scheme information strictly as available assistance, not guaranteed approval.")
        lines.append("2. Advise the farmer to verify exact eligibility and apply through official portals (e.g. FRUITS, Samrakshane, PM-KISAN) or the local Raitha Samparka Kendra (RSK).")
        lines.append("3. Do NOT invent eligibility conditions, subsidy percentages, or deadlines.")

        return "\n".join(lines).strip()
