"""
RaithaMitra Language Bridge Interface.

Defines the contract for bridging Kannada farmer queries with English/Hindi
agricultural LLMs (Dhenu2-1B, AgriParam, etc.).

Architecture Flow:
    Farmer Audio
         │
         ▼
    Kannada ASR  ──────────────>  Kannada Text
         │
         ▼
    Language Bridge  ──────────>  English Text
         │
         ▼
    Dhenu2 LLM  ───────────────>  English Advisory
         │
         ▼
    Language Bridge  ──────────>  Kannada Advisory Text (Future Phase)
         │
         ▼
    RaithaMitra API / UI
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict


class LanguageBridgeError(Exception):
    """Raised when translation or language bridging encounters an error."""
    pass


class LanguageBridge(ABC):
    """Abstract interface for cross-lingual translation bridge."""

    @abstractmethod
    def translate_to_advisory_lang(
        self,
        text: str,
        source_lang: str = "kn",
        target_lang: str = "en"
    ) -> str:
        """Translates query text from source language (Kannada) into advisory language (English/Hindi)."""
        pass

    @abstractmethod
    def translate_from_advisory_lang(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "kn"
    ) -> str:
        """Translates model response from advisory language back into Kannada."""
        pass

    def kannada_to_english(self, text: str) -> str:
        """Convenience alias for translating Kannada text to English."""
        return self.translate_to_advisory_lang(text, source_lang="kn", target_lang="en")

    def english_to_kannada(self, text: str) -> str:
        """Convenience alias for translating English text to Kannada."""
        return self.translate_from_advisory_lang(text, source_lang="en", target_lang="kn")


class PassThroughLanguageBridge(LanguageBridge):
    """
    Pass-through placeholder language bridge for testing and development.
    Preserves text without executing external translation models.
    """

    def translate_to_advisory_lang(
        self,
        text: str,
        source_lang: str = "kn",
        target_lang: str = "en"
    ) -> str:
        if not text or not text.strip():
            raise LanguageBridgeError("Cannot translate empty text.")
        return text.strip()

    def translate_from_advisory_lang(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "kn"
    ) -> str:
        if not text or not text.strip():
            raise LanguageBridgeError("Cannot translate empty text.")
        return text.strip()


class MockLanguageBridge(LanguageBridge):
    """
    Mock language bridge providing predefined sample translations for unit testing
    the Kannada -> English -> LLM -> Kannada pipeline without heavy translation weights.
    """

    SAMPLE_KN_TO_EN: Dict[str, str] = {
        "ನನ್ನ ಟೊಮೇಟೊ ಗಿಡದ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?": "What are common causes of yellow leaves in tomato plants, and what should a farmer check first?",
        "ಟೊಮೇಟೊ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ": "Tomato leaves are turning yellow. What should be done?",
        "ಭತ್ತದ ಬೆಳೆಗೆ ಯಾವ ಕೀಟನಾಶಕ ಬಳಸಬೇಕು?": "What pesticide should be used for paddy crop?",
        "ರಾಗಿ ಬಿತ್ತನೆಗೆ ಯಾವ ಕಾಲ ಸೂಕ್ತ?": "What is the suitable season for sowing ragi?"
    }

    SAMPLE_EN_TO_KN: Dict[str, str] = {
        "What are common causes of yellow leaves in tomato plants, and what should a farmer check first?": "ಟೊಮೇಟೊ ಗಿಡಗಳಲ್ಲಿ ಹಳದಿ ಎಲೆಗಳಿಗೆ ಸಾಮಾನ್ಯ ಕಾರಣಗಳು ಪೋಷಕಾಂಶಗಳ ಕೊರತೆ ಅಥವಾ ಅತಿಯಾದ ನೀರಾವರಿ. ಮೊದಲು ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ."
    }

    def translate_to_advisory_lang(
        self,
        text: str,
        source_lang: str = "kn",
        target_lang: str = "en"
    ) -> str:
        if not text or not text.strip():
            raise LanguageBridgeError("Cannot translate empty text.")
        clean = text.strip()
        return self.SAMPLE_KN_TO_EN.get(clean, f"Translated: {clean}")

    def translate_from_advisory_lang(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "kn"
    ) -> str:
        if not text or not text.strip():
            raise LanguageBridgeError("Cannot translate empty text.")
        clean = text.strip()
        return self.SAMPLE_EN_TO_KN.get(clean, clean)
