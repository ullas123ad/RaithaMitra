"""
RaithaMitra — Canonical Crop Identity & Registry Module (Phase 5.6A)
===================================================================
Provides deterministic, lightweight canonical crop recognition, categorization,
Karnataka agro-climatic suitability classification, and normalization for 100+
agricultural crops based on a three-tier support model:
  1. SUPPORTED: Validated agricultural package-of-practices in local RAG corpus.
  2. RECOGNIZED BUT NOT SUPPORTED: Known crop identity with insufficient localized knowledge.
  3. UNSUPPORTED: Unmapped / non-agricultural / out of scope.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Path to the authoritative crop registry JSON
REGISTRY_PATH = Path(__file__).parent / "crop_registry.json"


def _load_crop_registry() -> Dict[str, Any]:
    """Loads the authoritative machine-readable crop registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"crops": {}, "categories": []}


_REGISTRY_DATA = _load_crop_registry()
_CROPS_DICT = _REGISTRY_DATA.get("crops", {})

# Canonical list of crops with verified RAG agricultural knowledge
SUPPORTED_CROPS: List[str] = [
    crop_id for crop_id, data in _CROPS_DICT.items()
    if data.get("support_status") == "supported" and data.get("rag_supported", False)
]

# Canonical list of crops recognized by name but lacking full local RAG knowledge
RECOGNIZED_UNSUPPORTED_CROPS: List[str] = [
    crop_id for crop_id, data in _CROPS_DICT.items()
    if data.get("support_status") == "recognized_not_supported"
]


def _build_canonical_map() -> Dict[str, Dict[str, str]]:
    """Builds alias-to-canonical mapping dictionary from crop registry."""
    mapping: Dict[str, Dict[str, str]] = {}
    for canonical_id, data in _CROPS_DICT.items():
        primary_kn = data.get("kannada_names", [""])[0] if data.get("kannada_names") else canonical_id
        
        # Add canonical name itself
        mapping[canonical_id.lower()] = {"canonical": canonical_id, "kannada": primary_kn}
        mapping[canonical_id.replace("_", " ").lower()] = {"canonical": canonical_id, "kannada": primary_kn}
        
        # Add all listed english names, kannada names, and aliases
        all_aliases = set(data.get("english_names", [])) | set(data.get("kannada_names", [])) | set(data.get("aliases", []))
        for alias in all_aliases:
            if alias and alias.strip():
                clean = alias.strip().lower()
                mapping[clean] = {"canonical": canonical_id, "kannada": primary_kn}
                # Also handle spacing variants
                if " " in clean:
                    mapping[clean.replace(" ", "")] = {"canonical": canonical_id, "kannada": primary_kn}
    return mapping


CROP_CANONICAL_MAP: Dict[str, Dict[str, str]] = _build_canonical_map()

# Pre-sorted list of aliases by length descending for longest-match matching
_SORTED_ALIASES = sorted(CROP_CANONICAL_MAP.keys(), key=len, reverse=True)


# Ambiguous crop terms that should prompt for clarification when standalone
AMBIGUOUS_CROP_TERMS: Dict[str, Dict[str, Any]] = {
    "ಮೆಣಸು": {
        "clarification_prompt_kn": "ದಯವಿಟ್ಟು ನೀವು ಯಾವ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ (ಉದಾಹರಣೆಗೆ: ಹಸಿ ಮೆಣಸಿನಕಾಯಿ ಅಥವಾ ಕರಿಮೆಣಸು).",
        "clarification_prompt_en": "Please clarify which crop you are referring to (e.g. green chilli or black pepper).",
        "candidates": ["chilli", "black_pepper"]
    },
    "pepper": {
        "clarification_prompt_kn": "ದಯವಿಟ್ಟು ನೀವು ಯಾವ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ (ಉದಾಹರಣೆಗೆ: ಹಸಿ ಮೆಣಸಿನಕಾಯಿ ಅಥವಾ ಕರಿಮೆಣಸು).",
        "clarification_prompt_en": "Please clarify which crop you are referring to (e.g. green chilli or black pepper).",
        "candidates": ["chilli", "black_pepper"]
    },
    "ತರಕಾರಿ": {
        "clarification_prompt_kn": "ದಯವಿಟ್ಟು ನೀವು ನಿರ್ದಿಷ್ಟವಾಗಿ ಯಾವ ತರಕಾರಿ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ (ಉದಾಹರಣೆಗೆ: ಟೊಮ್ಯಾಟೊ, ಈರುಳ್ಳಿ, ಆಲೂಗಡ್ಡೆ, ಬದನೆಕಾಯಿ).",
        "clarification_prompt_en": "Please specify which vegetable crop you are asking about (e.g. tomato, onion, potato, brinjal).",
        "candidates": ["tomato", "onion", "potato", "brinjal"]
    },
    "ಹಣ್ಣು": {
        "clarification_prompt_kn": "ದಯವಿಟ್ಟು ನೀವು ನಿರ್ದಿಷ್ಟವಾಗಿ ಯಾವ ಹಣ್ಣಿನ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ (ಉದಾಹರಣೆಗೆ: ಬಾಳೆ, ಮಾವು, ದಾಳಿಂಬೆ, ದ್ರಾಕ್ಷಿ, ಕಲ್ಲಂಗಡಿ).",
        "clarification_prompt_en": "Please specify which fruit crop you are asking about (e.g. banana, mango, pomegranate, grapes, watermelon).",
        "candidates": ["banana", "mango", "pomegranate", "grapes", "watermelon"]
    }
}


def get_crop_registry() -> Dict[str, Any]:
    """Returns the full machine-readable crop registry data."""
    return _REGISTRY_DATA


def get_crop_entry(crop_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Returns the registry entry for a given canonical crop or alias."""
    if not crop_name:
        return None
    canonical = normalize_crop_name(crop_name)
    if canonical and canonical in _CROPS_DICT:
        return _CROPS_DICT[canonical]
    return None


def get_supported_crops() -> List[str]:
    """Returns list of all canonical crop IDs with verified RAG support."""
    return list(SUPPORTED_CROPS)


def get_recognized_crops() -> List[str]:
    """Returns list of all recognized crop IDs (supported + recognized_unsupported)."""
    return list(_CROPS_DICT.keys())


def is_crop_supported(crop_name: Optional[str]) -> bool:
    """Returns True if the crop has verified RAG package of practices."""
    if not crop_name:
        return False
    canonical = normalize_crop_name(crop_name)
    if not canonical or canonical not in _CROPS_DICT:
        return False
    entry = _CROPS_DICT[canonical]
    return entry.get("support_status") == "supported" and entry.get("rag_supported", False)


def get_crop_support_status(crop_name: Optional[str]) -> str:
    """
    Returns the three-tier support status for a crop:
      - 'supported'
      - 'recognized_not_supported'
      - 'unsupported'
    """
    if not crop_name:
        return "unsupported"
    canonical = normalize_crop_name(crop_name)
    if not canonical or canonical not in _CROPS_DICT:
        return "unsupported"
    return _CROPS_DICT[canonical].get("support_status", "unsupported")


def get_crop_category(crop_name: Optional[str]) -> Optional[str]:
    """Returns the agricultural category for a crop (e.g. 'cereal', 'pulse', 'spice', 'fruit')."""
    entry = get_crop_entry(crop_name)
    if entry:
        return entry.get("category")
    return None


def get_karnataka_suitability(crop_name: Optional[str]) -> str:
    """
    Returns the Karnataka agro-climatic cultivation suitability classification:
      - 'KARNATAKA_RELEVANT': Extensively/traditionally cultivated in Karnataka.
      - 'KARNATAKA_CONDITIONALLY_SUITABLE': Suitable under specific agro-ecological conditions / intercropping.
      - 'KARNATAKA_LIMITED': Localized high-elevation pockets / non-traditional experimental.
      - 'KARNATAKA_NOT_RECOMMENDED': Ecologically unviable for open commercial cultivation in Karnataka.
      - 'UNKNOWN': Insufficient authoritative regional evidence.
    """
    entry = get_crop_entry(crop_name)
    if entry:
        return entry.get("karnataka_suitability", "UNKNOWN")
    return "UNKNOWN"


def get_crop_suitability_details(crop_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Returns detailed agro-climatic suitability parameters and notes for a crop."""
    entry = get_crop_entry(crop_name)
    if entry:
        return {
            "canonical_name": entry.get("canonical_name"),
            "karnataka_suitability": entry.get("karnataka_suitability", "UNKNOWN"),
            "primary_districts": entry.get("primary_districts", []),
            "agro_climatic_details": entry.get("agro_climatic_details", {}),
            "source_institutions": entry.get("source_institutions", [])
        }
    return None


def check_crop_ambiguity(text: str) -> Optional[Dict[str, Any]]:
    """
    Checks if a query contains standalone ambiguous terms (e.g. bare 'ಮೆಣಸು' or 'pepper'
    without clarifying adjectives like 'ಹಸಿ', 'ಕಪ್ಪು', 'black', 'chilli').
    Returns clarification metadata if ambiguous, else None.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower().strip()
    words = set(re.findall(r"[\w\u0C80-\u0CFF]+", text_lower))

    # Check for exact standalone ambiguous words
    for term, data in AMBIGUOUS_CROP_TERMS.items():
        if term in words or text_lower == term:
            # Check if disambiguating adjectives are present
            if term in ["ಮೆಣಸು", "pepper"]:
                if any(w in text_lower for w in ["ಕರಿಮೆಣಸು", "ಕಾಳುಮೆಣಸು", "ಕಪ್ಪು", "black", "ಹಸಿಮೆಣಸು", "ಕೆಂಪು", "ಬ್ಯಾಡಗಿ", "chilli", "chili", "bell", "capsicum", "ದಪ್ಪ"]):
                    continue
            return data
    return None


def normalize_crop_name(name: Optional[str]) -> Optional[str]:
    """
    Normalizes any crop string (English alias, transliteration, or Kannada script)
    to its canonical English identifier.
    Uses longest-match precedence to prevent substring collisions.
    """
    if not name or not str(name).strip():
        return None

    clean_name = str(name).strip().lower()
    
    # 1. Exact match in canonical map
    mapping = CROP_CANONICAL_MAP.get(clean_name)
    if mapping:
        return mapping["canonical"]

    # 2. Match against sorted aliases by length descending
    for alias in _SORTED_ALIASES:
        # For English-only aliases, use word boundary regex if short
        if alias.isascii() and len(alias) <= 4:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, clean_name):
                return CROP_CANONICAL_MAP[alias]["canonical"]
        else:
            if alias in clean_name:
                return CROP_CANONICAL_MAP[alias]["canonical"]

    return None


def detect_crop_from_text(text: str) -> Optional[str]:
    """
    Detects canonical crop mentioned in arbitrary text (Kannada or English).
    Searches longer alias phrases first to avoid partial substring collisions,
    and applies word-boundary checks for short ASCII terms.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # Search in order of longest alias first
    for alias in _SORTED_ALIASES:
        # For short English aliases (<= 4 chars), enforce word boundaries
        if alias.isascii() and len(alias) <= 4:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, text_lower):
                return CROP_CANONICAL_MAP[alias]["canonical"]
        else:
            if alias in text_lower:
                return CROP_CANONICAL_MAP[alias]["canonical"]

    return None


def resolve_canonical_crop(
    query: str,
    translated_query: Optional[str] = None,
    explicit_crop: Optional[str] = None
) -> Optional[str]:
    """
    Resolves the canonical crop identity with strict priority:
      1. Explicit `explicit_crop` argument (if valid).
      2. Detection in original raw `query` (e.g., native Kannada terms).
      3. Detection in `translated_query` (English translation).

    This ensures mistranslations in step 3 (e.g. 'ಮೆಣಸಿನಕಾಯಿ' -> 'cucumber')
    never override the true crop detected in step 2 ('chilli').
    """
    # 1. Explicit argument priority
    if explicit_crop:
        norm = normalize_crop_name(explicit_crop)
        if norm:
            return norm

    # 2. Original query priority (e.g. Kannada terms directly)
    if query:
        detected = detect_crop_from_text(query)
        if detected:
            return detected

    # 3. Translated query fallback
    if translated_query:
        detected = detect_crop_from_text(translated_query)
        if detected:
            return detected

    return None
