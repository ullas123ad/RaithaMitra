"""
RaithaMitra — Farmer Distress Detection Configuration & Signal Taxonomy
========================================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Non-Negotiable Principles:
1. Distress detection is a conversational safety signal, NOT a medical, psychiatric,
   or clinical diagnosis.
2. The system NEVER claims a farmer has depression, mental illness, or suicidal condition.
3. No emergency numbers or helplines are fabricated. For high-risk safety, the system
   guides the user to stay with trusted family/friends, visit the nearest medical clinic/hospital,
   or contact local emergency services.
4. For agricultural distress, official local support institutions (Raitha Samparka Kendra, KVK)
   are recommended.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class DistressLevel(str, Enum):
    """Three-tier conservative distress classification."""
    NONE = "NONE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


@dataclass
class DistressResult:
    """Structured result from distress detection analysis."""
    level: DistressLevel
    detected: bool
    priority: str  # 'normal', 'advisory', 'safety'
    trigger_categories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    empathy_message_kn: Optional[str] = None
    empathy_message_en: Optional[str] = None
    safety_message_kn: Optional[str] = None
    safety_message_en: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """Returns safe, clean dictionary for public API payload without exposing internal keywords."""
        return {
            "detected": self.detected,
            "level": self.level.value,
            "priority": self.priority
        }


# =============================================================================
# VERIFIED RESPONSE TEMPLATES (NO INVENTED PHONE NUMBERS)
# =============================================================================

# HIGH Severity Safety Message (Calm, urgent, non-judgmental, immediate human action)
SAFETY_RESPONSE_KN = (
    "ನೀವು ಈಗ ತುಂಬಾ ಕಷ್ಟದಲ್ಲಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಒಬ್ಬರೇ ಇರಬೇಡಿ. "
    "ನಿಮ್ಮ ಕುಟುಂಬದವರು ಅಥವಾ ವಿಶ್ವಾಸದ ವ್ಯಕ್ತಿಯೊಬ್ಬರನ್ನು ಈಗಲೇ ಸಂಪರ್ಕಿಸಿ ಮತ್ತು "
    "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ಅಥವಾ ಸ್ಥಳೀಯ ತುರ್ತು ಸೇವೆಯಿಂದ ತಕ್ಷಣ ಸಹಾಯ ಪಡೆಯಿರಿ."
)

SAFETY_RESPONSE_EN = (
    "It sounds like you are going through an extremely difficult situation. Please do not stay alone. "
    "Reach out to a family member, trusted person, or your nearest hospital/emergency services immediately for support."
)

# MODERATE Severity Empathy Prefix
EMPATHY_PREFIX_KN = "ನಿಮ್ಮ ಪರಿಸ್ಥಿತಿ ಮತ್ತು ಕಷ್ಟ ನಮಗೆ ಅರ್ಥವಾಗುತ್ತದೆ. ಧೈರ್ಯವಾಗಿರಿ. "
EMPATHY_PREFIX_EN = "We understand that this is a difficult situation. Please stay strong. "

# MODERATE Severity Agricultural Support Referral
AGRICULTURAL_SUPPORT_REFERRAL_KN = (
    "ಇದರ ಜೊತೆಗೆ ನಿಮ್ಮ ಕುಟುಂಬ ಅಥವಾ ವಿಶ್ವಾಸದ ವ್ಯಕ್ತಿಯೊಂದಿಗೆ ಮಾತನಾಡಿ. ಹೆಚ್ಚಿನ ಕೃಷಿ ಸಲಹೆ ಹಾಗೂ "
    "ಸರ್ಕಾರದ ಪರಿಹಾರ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರ (RSK) ಅಥವಾ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು (KVK) ಸಂಪರ್ಕಿಸಿ."
)

AGRICULTURAL_SUPPORT_REFERRAL_EN = (
    "Please also share your concerns with family or a trusted friend. For localized agricultural guidance "
    "and government relief procedures, visit your nearest Raitha Samparka Kendra (RSK) or Krishi Vigyan Kendra (KVK)."
)


# =============================================================================
# KANNADA, ENGLISH & MIXED SIGNAL TAXONOMIES
# =============================================================================

# 1. Plant / Crop Objects (Used to ensure plant death is not confused with human distress)
PLANT_OBJECT_TERMS: Set[str] = {
    # Kannada
    "ಬೆಳೆ", "ಬೆಳೆಯ", "ಬೆಳೆಗೆ", "ಗಿಡ", "ಗಿಡಗಳು", "ಗಿಡದ", "ಮರ", "ಮರಗಳು", "ಸಸಿ", "ಸಸಿಗಳು",
    "ಎಲೆ", "ಎಲೆಗಳು", "ಹೂ", "ಹೂವು", "ಕಾಯಿ", "ಕಾಳು", "ತೋಟ", "ಗದ್ದೆ", "ಹೊಲ", "ಕಟಾವು",
    "ರಾಗಿ", "ಭತ್ತ", "ಮೆಕ್ಕೆಜೋಳ", "ಜೋಳ", "ಸಜ್ಜೆ", "ತೊಗರಿ", "ಕಡಲೆ", "ಹೆಸರು", "ಉದ್ದು",
    "ಕಡಲೆಕಾಯಿ", "ಶೇಂಗಾ", "ಸೂರ್ಯಕಾಂತಿ", "ಸೋಯಾಬೀನ್", "ಮೆಣಸಿನಕಾಯಿ", "ಅರಿಶಿನ", "ಶುಂಠಿ",
    "ಕರಿಮೆಣಸು", "ಏಲಕ್ಕಿ", "ಟೊಮ್ಯಾಟೊ", "ಟೊಮೇಟೊ", "ಈರುಳ್ಳಿ", "ಆಲೂಗಡ್ಡೆ", "ಬದನೆಕಾಯಿ",
    "ಎಲೆಕೋಸು", "ಬೆಂಡೆಕಾಯಿ", "ಬಾಳೆ", "ಮಾವು", "ದಾಳಿಂಬೆ", "ದ್ರಾಕ್ಷಿ", "ಪಪ್ಪಾಯಿ", "ನಿಂಬೆ",
    "ಕಲ್ಲಂಗಡಿ", "ಅಡಿಕೆ", "ತೆಂಗು", "ಕಾಫಿ", "ಗೋಡಂಬಿ", "ಕಬ್ಬು", "ಹತ್ತಿ", "ತಂಬಾಕು", "ಮಲ್ಲಿಗೆ", "ಚೆಂಡುಹೂ",
    # English
    "crop", "crops", "plant", "plants", "tree", "trees", "leaf", "leaves", "flower",
    "flowers", "fruit", "fruits", "field", "fields", "farm", "paddy", "ragi", "maize",
    "wheat", "tomato", "chilli", "onion", "potato", "cotton", "sugarcane", "arecanut",
    "banana", "mango", "watermelon", "grapes", "coffee", "plantation", "harvest"
}

# 2. Plant Damage / Agricultural Symptom Verbs
PLANT_DAMAGE_VERBS: Set[str] = {
    "ಸತ್ತುಹೋಗುತ್ತಿದೆ", "ಸತ್ತುಹೋಗಿದೆ", "ಸಾಯುತ್ತಿದೆ", "ಸತ್ತಿದೆ", "ಒಣಗುತ್ತಿದೆ", "ಒಣಗಿದೆ",
    "ಕೊಳೆಯುತ್ತಿದೆ", "ಕೊಳೆತಿದೆ", "ಹಾಳಾಗಿದೆ", "ನಾಶವಾಗಿದೆ", "ಉದುರುತ್ತಿದೆ", "ಮುದುರುತ್ತಿದೆ",
    "ಬಾಡುತ್ತಿದೆ", "ಬಾಡಿದೆ", "ಹಳದಿಯಾಗುತ್ತಿದೆ", "ಕಪ್ಪಾಗುತ್ತಿದೆ", "ರೋಗ ಬಂದಿದೆ",
    "dying", "dead", "drying", "rotting", "damaged", "destroyed", "wilting", "yellowing", "dropping"
}

# 3. HIGH Severity Immediate Danger / Self-Harm Signals (Direct personal human intent)
HIGH_SEVERITY_SIGNALS_KN: List[str] = [
    "ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು",
    "ಜೀವನ ಮುಗಿಸುತ್ತೇನೆ",
    "ಜೀವನ ಮುಗಿಸಬೇಕು",
    "ಜೀವ ಕಳೆದುಕೊಳ್ಳಬೇಕು",
    "ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ",
    "ಬದುಕಲು ಇಷ್ಟಪಡುವುದಿಲ್ಲ",
    "ಬದುಕಲು ಸಾಧ್ಯವಿಲ್ಲ ಸಾಯಬೇಕು",
    "ಸಾಯಬೇಕು ಅನ್ನಿಸುತ್ತಿದೆ",
    "ಸಾಯಲು ಬಯಸುತ್ತೇನೆ",
    "ಆತ್ಮಹತ್ಯೆ ಮಾಡಿಕೊಳ್ಳಬೇಕು",
    "ಆತ್ಮಹತ್ಯೆ ಮಾಡಿಕೊಳ್ಳುತ್ತೇನೆ",
    "ಆತ್ಮಹತ್ಯೆ ಮಾಡಿಕೊಳ್ಳುವೆ",
    "ನನ್ನ ಜೀವನ ಮುಗಿದಂತಾಗಿದೆ ಬದುಕುವುದಿಲ್ಲ",
    "ಇನ್ನು ನಾನು ಬದುಕುವುದಿಲ್ಲ",
    "ನಾನು ಸಾಯುತ್ತೇನೆ",
    "ನಾನು ಸಾಯಬೇಕು",
    "ಜೀವ ಕಳೆದುಕೊಳ್ಳುತ್ತೇನೆ",
    "ಪ್ರಾಣ ಕಳೆದುಕೊಳ್ಳುತ್ತೇನೆ",
    "ಪ್ರಾಣ ಬಿಡುತ್ತೇನೆ",
    "ಜೀವನವೇ ಬೇಡವಾಗಿದೆ ಸಾಯಬೇಕು"
]

HIGH_SEVERITY_SIGNALS_EN: List[str] = [
    "want to end my life",
    "going to end my life",
    "want to die",
    "going to die by suicide",
    "commit suicide",
    "committing suicide",
    "do not want to live anymore",
    "don't want to live anymore",
    "cannot live anymore",
    "feel like killing myself",
    "feel like ending my life",
    "i will kill myself",
    "i am going to kill myself",
    "better off dead"
]

# 4. MODERATE Severity Distress Signals (Financial distress, debt crisis, severe emotional burden)
DEBT_FINANCIAL_SIGNALS_KN: List[str] = [
    "ಸಾಲ ತೀರಿಸಲು ಆಗುತ್ತಿಲ್ಲ",
    "ಸಾಲ ತೀರಿಸಲು ಆಗುವುದಿಲ್ಲ",
    "ಸಾಲ ತೀರಿಸೋಕೆ ಆಗಲ್ಲ",
    "ಸಾಲ ತೀರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ",
    "ಸಾಲ ತೀರಿಸಲಾಗುತ್ತಿಲ್ಲ",
    "ಸಾಲ ಕಟ್ಟಲು ಆಗುತ್ತಿಲ್ಲ",
    "ಸಾಲ ಕಟ್ಟಲು ಆಗುವುದಿಲ್ಲ",
    "ಸಾಲದ ಒತ್ತಡ ತುಂಬಾ ಇದೆ",
    "ಸಾಲದ ಹೊರೆ",
    "ಸಾಲ ತುಂಬಾ ಇದೆ",
    "ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ",
    "ಸಾಲ ತೀರಿಸೋದು ಹೇಗೆ",
    "ಬ್ಯಾಂಕ್ ಸಾಲ ಕಟ್ಟಲು ಆಗುತ್ತಿಲ್ಲ",
    "ಲಕ್ಷ ಸಾಲ ಇದೆ",
    "ಸಾಲಗಾರರು ಕಾಟ ಕೊಡುತ್ತಿದ್ದಾರೆ",
    "ಸಾಲದ ಸಮಸ್ಯೆ"
]

DEBT_FINANCIAL_SIGNALS_EN: List[str] = [
    "cannot repay loan",
    "unable to repay loan",
    "cannot pay back debt",
    "loan pressure",
    "heavy debt burden",
    "too much debt",
    "how to repay loan",
    "financial crisis",
    "cannot manage loan",
    "bank loan pressure"
]

SEVERE_LOSS_HOPELESSNESS_KN: List[str] = [
    "ತುಂಬಾ ಕಷ್ಟವಾಗುತ್ತಿದೆ",
    "ಬದುಕು ಕಷ್ಟವಾಗಿದೆ",
    "ಜೀವನ ಕಷ್ಟವಾಗಿದೆ",
    "ಏನು ಮಾಡಬೇಕು ಎಂದು ಗೊತ್ತಾಗುತ್ತಿಲ್ಲ",
    "ಏನು ಮಾಡಬೇಕು ಗೊತ್ತಾಗುತ್ತಿಲ್ಲ",
    "ಏನು ಮಾಡೋದು ತಿಳಿಯುತ್ತಿಲ್ಲ",
    "ಎಲ್ಲವೂ ನಷ್ಟವಾಗಿದೆ",
    "ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ ಮತ್ತು ಸಾಲ",
    "ಸಂಪೂರ್ಣ ಬೆಳೆ ನಷ್ಟವಾಗಿದೆ",
    "ಎಲ್ಲವೂ ಹಾಳಾಗಿದೆ",
    "ತುಂಬಾ ಸಂಕಷ್ಟದಲ್ಲಿದ್ದೇನೆ",
    "ಎರಡು ವರ್ಷಗಳಿಂದ ಬೆಳೆ ನಷ್ಟ",
    "ಸತತ ಬೆಳೆ ನಷ್ಟ",
    "ಭಾರೀ ನಷ್ಟವಾಗಿದೆ",
    "ಜೀವನವೇ ಕತ್ತಲೆಯಾಗಿದೆ"
]

SEVERE_LOSS_HOPELESSNESS_EN: List[str] = [
    "very difficult to survive",
    "don't know what to do",
    "do not know what to do",
    "everything is lost",
    "completely ruined",
    "severe financial loss",
    "too much stress",
    "heavy loss for two years",
    "crop loss for two years",
    "loss for two years",
    "repeated crop failure",
    "continuous crop failure",
    "suffered crop loss",
    "extremely difficult",
    "struggling to survive"
]

# Mixed Language Codeswitching Signals
MIXED_CODE_SWITCHING_SIGNALS: List[str] = [
    "crop full loss ಆಗಿದೆ",
    "loan ಹೇಗೆ pay ಮಾಡಲಿ",
    "loan pay ಮಾಡಲು ಆಗಲ್ಲ",
    "loan pay ಮಾಡೋಕೆ ಆಗಲ್ಲ",
    "ತುಂಬಾ stress ಆಗಿದೆ",
    "full tension ಆಗಿದೆ",
    "heavy loss ಆಗಿದೆ",
    "loan pressure ತುಂಬಾ ಇದೆ",
    "cannot manage ಸಾಲ",
    "totally lost ಆಗಿದೆ"
]


@dataclass
class DistressConfig:
    """Distress detector runtime configuration."""
    enabled: bool = True
    fast_path_for_high: bool = True
    confidence_threshold_moderate: float = 0.5
    confidence_threshold_high: float = 0.8
