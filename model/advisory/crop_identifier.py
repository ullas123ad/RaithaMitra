"""
RaithaMitra — Canonical Crop Identity Module
=============================================
Provides deterministic, lightweight canonical crop recognition and normalization
for Karnataka agriculture. Ensures Kannada crop terms and English aliases are mapped
to canonical crop identities, taking strict precedence over general translation variance.
"""

from typing import Dict, List, Optional, Set, Tuple
import re

# Supported canonical crops in RaithaMitra
SUPPORTED_CROPS: List[str] = [
    "ragi",
    "paddy",
    "maize",
    "groundnut",
    "sugarcane",
    "cotton",
    "chilli",
    "onion",
    "potato",
    "banana",
    "tomato",
]

# Canonical crop metadata with Kannada script names and standard aliases
CROP_CANONICAL_MAP: Dict[str, Dict[str, str]] = {
    # Ragi / Finger Millet
    "ragi": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "finger millet": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "fingermillet": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "mandua": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "eleusine coracana": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "ರಾಗಿ": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "ರಾಗಿಯ": {"canonical": "ragi", "kannada": "ರಾಗಿ"},
    "ರಾಗಿಬೆಳೆ": {"canonical": "ragi", "kannada": "ರಾಗಿ"},

    # Paddy / Rice
    "paddy": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "rice": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "paddy crop": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "bhatta": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "oryza sativa": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "ಭತ್ತ": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "ಭತ್ತದ": {"canonical": "paddy", "kannada": "ಭತ್ತ"},
    "ಭತ್ತಬೆಳೆ": {"canonical": "paddy", "kannada": "ಭತ್ತ"},

    # Maize / Corn
    "maize": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "corn": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "makka": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "mekkejola": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "zea mays": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "ಮೆಕ್ಕೆಜೋಳ": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "ಮೆಕ್ಕೆ ಜೋಳ": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},
    "ಮೆಕ್ಕೆಜೋಳದ": {"canonical": "maize", "kannada": "ಮೆಕ್ಕೆಜೋಳ"},

    # Groundnut / Peanut
    "groundnut": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "peanut": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "kadlekai": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "shenga": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "arachis hypogaea": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "ಕಡಲೆಕಾಯಿ": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "ಕಡಲೆ ಕಾಯಿ": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "ಕಡಲೆಕಾಯಿಯ": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},
    "ಶೇಂಗಾ": {"canonical": "groundnut", "kannada": "ಕಡಲೆಕಾಯಿ"},

    # Sugarcane
    "sugarcane": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "sugar cane": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "cane": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "kabbu": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "saccharum officinarum": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "ಕಬ್ಬು": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "ಕಬ್ಬಿನ": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},
    "ಕಬ್ಬಿನ ಬೆಳೆ": {"canonical": "sugarcane", "kannada": "ಕಬ್ಬು"},

    # Cotton
    "cotton": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "kapas": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "hatti": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "gossypium": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "ಹತ್ತಿ": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "ಹತ್ತಿಯ": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},
    "ಹತ್ತಿ ಬೆಳೆ": {"canonical": "cotton", "kannada": "ಹತ್ತಿ"},

    # Chilli
    "chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "chili": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "green chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "red chilli": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "mirchi": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "menasinakai": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "menasinakayi": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "capsicum": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "ಮೆಣಸಿನಕಾಯಿ": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "ಮೆಣಸಿನ ಕಾಯಿ": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "ಮೆಣಸಿನಕಾಯಿಯ": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},
    "ಮೆಣಸಿನಗಿಡ": {"canonical": "chilli", "kannada": "ಮೆಣಸಿನಕಾಯಿ"},

    # Onion
    "onion": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "eerulli": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "pyaz": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "allium cepa": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "ಈರುಳ್ಳಿ": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "ಈರುಳ್ಳಿಯ": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},
    "ಈರುಳ್ಳಿ ಬೆಳೆ": {"canonical": "onion", "kannada": "ಈರುಳ್ಳಿ"},

    # Potato
    "potato": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "aalugadde": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "aloo": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "solanum tuberosum": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "ಆಲೂಗಡ್ಡೆ": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "ಆಲೂಗೆಡ್ಡೆ": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},
    "ಆಲೂಗಡ್ಡೆಯ": {"canonical": "potato", "kannada": "ಆಲೂಗಡ್ಡೆ"},

    # Banana
    "banana": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "plantain": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "baale": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "bale": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "musa": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "ಬಾಳೆ": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "ಬಾಳೆಯ": {"canonical": "banana", "kannada": "ಬಾಳೆ"},
    "ಬಾಳೆಹಣ್ಣು": {"canonical": "banana", "kannada": "ಬಾಳೆ"},

    # Tomato
    "tomato": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "tamota": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "tamatar": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "tometo": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "solanum lycopersicum": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮ್ಯಾಟೊ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮೆಟೊ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮೇಟೊ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
    "ಟೊಮ್ಯಾಟೊದ": {"canonical": "tomato", "kannada": "ಟೊಮ್ಯಾಟೊ"},
}


def normalize_crop_name(name: Optional[str]) -> Optional[str]:
    """
    Normalizes any crop string (English alias, transliteration, or Kannada script)
    to its canonical English identifier.

    Args:
        name: Raw crop name string.

    Returns:
        Canonical crop string (e.g. 'ragi', 'paddy', 'chilli') or None if unmapped.
    """
    if not name or not str(name).strip():
        return None

    clean_name = str(name).strip().lower()
    mapping = CROP_CANONICAL_MAP.get(clean_name)
    if mapping:
        return mapping["canonical"]

    # Try word-boundary / substring matching against aliases
    for alias, meta in CROP_CANONICAL_MAP.items():
        if alias in clean_name:
            return meta["canonical"]

    return None


def detect_crop_from_text(text: str) -> Optional[str]:
    """
    Detects canonical crop mentioned in arbitrary text (Kannada or English).
    Searches longer alias phrases first to avoid partial substring collisions.

    Args:
        text: Input string.

    Returns:
        Canonical crop name if detected, else None.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # Sort aliases by length descending so multi-word aliases match before single words
    sorted_aliases = sorted(CROP_CANONICAL_MAP.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        # Check direct inclusion
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

    Args:
        query: Original farmer query (Kannada or English).
        translated_query: Translated English query if available.
        explicit_crop: Explicitly passed crop name string.

    Returns:
        Canonical crop identifier (e.g. 'chilli', 'onion', 'ragi') or None.
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
