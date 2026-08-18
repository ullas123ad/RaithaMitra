"""
RaithaMitra Language Bridge Interface.

Defines the contract for bridging Kannada farmer queries with AgriParam's
supported native languages (English and Hindi).

Architecture Flow:
    Farmer Audio
         │
         ▼
    Kannada ASR  ──────────────>  Kannada Text
         │
         ▼
    Language Bridge  ──────────>  English / Hindi Text
         │
         ▼
    AgriParam LLM  ────────────>  English / Hindi Advisory
         │
         ▼
    Language Bridge  ──────────>  Kannada Advisory Text
         │
         ▼
    RaithaMitra API / UI
"""

from abc import ABC, abstractmethod
from typing import Optional


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
        """
        Translates query text from source language (Kannada) into an AgriParam supported language (English/Hindi).

        Args:
            text: Query text in source language.
            source_lang: Language code of input (default: "kn" for Kannada).
            target_lang: Language code of target (default: "en" or "hi").

        Returns:
            Translated query text.
        """
        pass

    @abstractmethod
    def translate_from_advisory_lang(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "kn"
    ) -> str:
        """
        Translates model response from AgriParam language (English/Hindi) back into Kannada.

        Args:
            text: Advisory response in English/Hindi.
            source_lang: Language code of model output ("en" or "hi").
            target_lang: Target language code ("kn" for Kannada).

        Returns:
            Translated advisory response in Kannada.
        """
        pass


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
