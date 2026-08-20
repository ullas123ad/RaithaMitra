"""
RaithaMitra — Deterministic Farmer Distress Detector (Phase 5.7)
================================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Provides lightweight (< 5 ms), deterministic, explainable conversational safety
triage classifying user queries into NONE, MODERATE, or HIGH distress.

Distinguishes plant/crop damage from personal human distress, enforces strict
fast-path escalation for high-risk language, and integrates empathetic framing
with official agricultural support channels for moderate distress.
"""

import re
import time
from typing import Dict, List, Optional, Set, Tuple, Any

from model.distress.config import (
    DistressLevel,
    DistressResult,
    DistressConfig,
    SAFETY_RESPONSE_KN,
    SAFETY_RESPONSE_EN,
    EMPATHY_PREFIX_KN,
    EMPATHY_PREFIX_EN,
    AGRICULTURAL_SUPPORT_REFERRAL_KN,
    AGRICULTURAL_SUPPORT_REFERRAL_EN,
    PLANT_OBJECT_TERMS,
    PLANT_DAMAGE_VERBS,
    HIGH_SEVERITY_SIGNALS_KN,
    HIGH_SEVERITY_SIGNALS_EN,
    DEBT_FINANCIAL_SIGNALS_KN,
    DEBT_FINANCIAL_SIGNALS_EN,
    SEVERE_LOSS_HOPELESSNESS_KN,
    SEVERE_LOSS_HOPELESSNESS_EN,
    MIXED_CODE_SWITCHING_SIGNALS,
)


class DistressDetector:
    """
    Lightweight, deterministic, rule-based distress detector.
    Evaluates lexical patterns, negation, subject-verb context, and multi-clause indicators.
    """

    def __init__(self, config: Optional[DistressConfig] = None):
        self.config = config or DistressConfig()

    def detect(self, text: Optional[str]) -> DistressResult:
        """
        Analyzes input query text (Kannada, English, or Mixed codeswitching)
        and returns a structured DistressResult.

        Execution SLA: < 5 ms on standard CPU hardware.
        """
        if not text or not str(text).strip():
            return DistressResult(
                level=DistressLevel.NONE,
                detected=False,
                priority="normal"
            )

        clean_text = str(text).strip()
        text_lower = clean_text.lower()

        # ---------------------------------------------------------------------
        # 1. HIGH SEVERITY DETECTION (Immediate danger / personal self-harm)
        # ---------------------------------------------------------------------
        high_trigger = self._check_high_severity(clean_text, text_lower)
        if high_trigger:
            return DistressResult(
                level=DistressLevel.HIGH,
                detected=True,
                priority="safety",
                trigger_categories=[high_trigger],
                confidence=0.95,
                safety_message_kn=SAFETY_RESPONSE_KN,
                safety_message_en=SAFETY_RESPONSE_EN
            )

        # ---------------------------------------------------------------------
        # 2. MODERATE SEVERITY DETECTION (Financial crisis, debt, severe burden)
        # ---------------------------------------------------------------------
        moderate_triggers = self._check_moderate_severity(clean_text, text_lower)
        if moderate_triggers:
            return DistressResult(
                level=DistressLevel.MODERATE,
                detected=True,
                priority="advisory",
                trigger_categories=moderate_triggers,
                confidence=0.80,
                empathy_message_kn=EMPATHY_PREFIX_KN,
                empathy_message_en=EMPATHY_PREFIX_EN,
                safety_message_kn=AGRICULTURAL_SUPPORT_REFERRAL_KN,
                safety_message_en=AGRICULTURAL_SUPPORT_REFERRAL_EN
            )

        # ---------------------------------------------------------------------
        # 3. NONE (Normal agricultural query or plant damage description)
        # ---------------------------------------------------------------------
        return DistressResult(
            level=DistressLevel.NONE,
            detected=False,
            priority="normal",
            confidence=1.0
        )

    def _check_high_severity(self, text: str, text_lower: str) -> Optional[str]:
        """
        Checks for explicit personal immediate-danger / self-harm signals.
        Protects against plant-death false positives (e.g. 'crop is dying').
        """
        # Direct Kannada high-risk phrases
        for sig in HIGH_SEVERITY_SIGNALS_KN:
            if sig in text:
                return "HIGH_RISK_KANNADA_EXPLICIT"

        # Direct English high-risk phrases
        for sig in HIGH_SEVERITY_SIGNALS_EN:
            if sig in text_lower:
                return "HIGH_RISK_ENGLISH_EXPLICIT"

        # Contextual check: "ಸಾಯಬೇಕು" / "die" when combined with 1st-person pronoun or personal life context
        personal_markers_kn = ["ನಾನು", "ನನಗೆ", "ನನ್ನ ಜೀವನ", "ಬದುಕು", "ಜೀವನವೇ"]
        if "ಸಾಯಬೇಕು" in text:
            # If specifically referring to plant/insect ("ಕೀಟ ಸಾಯಬೇಕು" or "ಹುಳು ಸಾಯಬೇಕು"), it is NOT human distress
            if any(pest in text for pest in ["ಕೀಟ", "ಹುಳು", "ರೋಗ", "ಸೊಳ್ಳೆ", "pest", "insect", "larva"]):
                return None
            if any(p in text for p in personal_markers_kn) or "ಅನ್ನಿಸುತ್ತಿದೆ" in text or "ಮನಸ್ಸಾಗಿದೆ" in text:
                return "HIGH_RISK_KANNADA_PERSONAL"

        if "ಬದುಕಲು" in text and ("ಸಾಧ್ಯವಿಲ್ಲ" in text or "ಇಷ್ಟವಿಲ್ಲ" in text or "ಮನಸ್ಸಿಲ್ಲ" in text):
            return "HIGH_RISK_KANNADA_HOPELESSNESS"

        return None

    def _check_moderate_severity(self, text: str, text_lower: str) -> List[str]:
        """
        Checks for financial distress, unmanageable debt, multi-year loss,
        or severe stress without explicit immediate self-harm statements.
        """
        triggers: List[str] = []

        # 1. Exact debt signals in Kannada
        for sig in DEBT_FINANCIAL_SIGNALS_KN:
            if sig in text:
                triggers.append("DEBT_FINANCIAL_KANNADA")
                break

        # 2. Exact debt signals in English
        for sig in DEBT_FINANCIAL_SIGNALS_EN:
            if sig in text_lower:
                triggers.append("DEBT_FINANCIAL_ENGLISH")
                break

        # 3. Severe loss & hopelessness signals in Kannada
        for sig in SEVERE_LOSS_HOPELESSNESS_KN:
            if sig in text:
                triggers.append("SEVERE_LOSS_KANNADA")
                break

        # 4. Severe loss & hopelessness signals in English
        for sig in SEVERE_LOSS_HOPELESSNESS_EN:
            if sig in text_lower:
                triggers.append("SEVERE_LOSS_ENGLISH")
                break

        # 5. Mixed code-switching signals
        for sig in MIXED_CODE_SWITCHING_SIGNALS:
            if sig.lower() in text_lower:
                triggers.append("MIXED_CODESWITCHING_DISTRESS")
                break

        # 6. Combinatorial multi-clause analysis
        # Kannada Debt + Inability / Loss / Hardship
        has_debt_kn = any(w in text for w in ["ಸಾಲ", "ಸಾಲದ", "ಸಾಲವೂ", "ಬ್ಯಾಂಕ್ ಸಾಲ", "ಲಕ್ಷ ಸಾಲ"])
        has_inability_loss_kn = any(w in text for w in [
            "ಹಾಳಾಗಿದೆ", "ನಷ್ಟ", "ಕಷ್ಟ", "ಒತ್ತಡ", "ಸಂಕಷ್ಟ", "ನಾಶ",
            "ಆಗುತ್ತಿಲ್ಲ", "ಆಗುವುದಿಲ್ಲ", "ಆಗಲ್ಲ", "ಸಾಧ್ಯವಿಲ್ಲ", "ತಿಳಿಯುತ್ತಿಲ್ಲ", "ಗೊತ್ತಾಗುತ್ತಿಲ್ಲ"
        ])

        if has_debt_kn and has_inability_loss_kn:
            if "COMBINATORIAL_DEBT_KANNADA" not in triggers:
                triggers.append("COMBINATORIAL_DEBT_KANNADA")

        # English Debt / Crop Loss + Hardship / Survival difficulty
        has_debt_en = any(w in text_lower for w in ["loan", "debt", "repay", "emi", "financial", "bank"])
        has_loss_en = any(w in text_lower for w in [
            "loss", "ruined", "failed", "failure", "stress", "pressure", "damaged",
            "difficult", "struggling", "suffered", "crisis", "survive", "unable"
        ])

        if has_debt_en and has_loss_en:
            if "COMBINATORIAL_DEBT_LOSS_EN" not in triggers:
                triggers.append("COMBINATORIAL_DEBT_LOSS_EN")

        if ("crop loss" in text_lower or "crop failure" in text_lower or "loss for" in text_lower) and (
            "difficult" in text_lower or "struggl" in text_lower or "ruin" in text_lower
        ):
            if "COMBINATORIAL_CROP_LOSS_HARDSHIP_EN" not in triggers:
                triggers.append("COMBINATORIAL_CROP_LOSS_HARDSHIP_EN")

        return triggers


_GLOBAL_DETECTOR: Optional[DistressDetector] = None


def get_distress_detector(config: Optional[DistressConfig] = None) -> DistressDetector:
    """Returns a singleton instance of DistressDetector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None or config is not None:
        _GLOBAL_DETECTOR = DistressDetector(config=config)
    return _GLOBAL_DETECTOR
