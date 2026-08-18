"""
RaithaMitra Language Bridge Interface and Implementations.

Provides cross-lingual translation between Kannada farmer queries and English/Hindi
agricultural LLMs (Dhenu2-1B, AgriParam, etc.) using:
1. PassThroughLanguageBridge: For testing without translation.
2. MockLanguageBridge: Deterministic sample mappings for fast unit tests.
3. NLLBTranslationBridge: Free, local, CPU-based bidirectional translation
   using facebook/nllb-200-distilled-600M.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import os


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


class NLLBTranslationBridge(LanguageBridge):
    """
    Local CPU bidirectional translation bridge using facebook/nllb-200-distilled-600M.
    Supports Kannada (kan_Knda) <-> English (eng_Latn) <-> Hindi (hin_Deva).
    Uses strict lazy loading.
    """

    LANG_MAP: Dict[str, str] = {
        "kn": "kan_Knda",
        "en": "eng_Latn",
        "hi": "hin_Deva"
    }

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir or os.getenv("HF_HOME", None)
        self._tokenizer = None
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if the translation model is loaded in memory."""
        return self._is_loaded

    def load_model(self) -> None:
        """Lazily loads the NLLB model and tokenizer on CPU."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir
            )
            self._model.to(self.device)
            self._model.eval()

            torch.set_num_threads(4)
            self._is_loaded = True

        except Exception as e:
            raise LanguageBridgeError(f"Failed to load NLLB translation model: {str(e)}")

    def _translate(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 200) -> str:
        if not text or not text.strip():
            raise LanguageBridgeError("Cannot translate empty text.")

        if src_lang == tgt_lang:
            return text.strip()

        if not self._is_loaded:
            self.load_model()

        try:
            import torch

            src_code = self.LANG_MAP.get(src_lang, src_lang)
            tgt_code = self.LANG_MAP.get(tgt_lang, tgt_lang)

            self._tokenizer.src_lang = src_code
            inputs = self._tokenizer(text.strip(), return_tensors="pt").to(self.device)
            tgt_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=max_new_tokens
                )

            translated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text.strip()

        except Exception as e:
            raise LanguageBridgeError(f"NLLB translation error ({src_lang}->{tgt_lang}): {str(e)}")

    def translate_to_advisory_lang(
        self,
        text: str,
        source_lang: str = "kn",
        target_lang: str = "en"
    ) -> str:
        return self._translate(text, src_lang=source_lang, tgt_lang=target_lang, max_new_tokens=150)

    def translate_from_advisory_lang(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "kn"
    ) -> str:
        return self._translate(text, src_lang=source_lang, tgt_lang=target_lang, max_new_tokens=250)
