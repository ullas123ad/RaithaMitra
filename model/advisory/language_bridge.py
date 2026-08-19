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
import re


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
        "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ. ಏನು ಮಾಡಬೇಕು?": "There has been very little rain and my ragi crop is drying. What should I do?",
        "ಭತ್ತದ ಬೆಳೆಗೆ ಯಾವ ಕೀಟನಾಶಕ ಬಳಸಬೇಕು?": "What pesticide should be used for paddy crop?",
        "ರಾಗಿ ಬಿತ್ತನೆಗೆ ಯಾವ ಕಾಲ ಸೂಕ್ತ?": "What is the suitable season for sowing ragi?",
        "ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?": "My chilli plant leaves are curling. What should I do?",
        "ನನ್ನ ಈರುಳ್ಳಿ ಬೆಳೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ಏನು ಮಾಡಬೇಕು?": "My onion crop is getting too much rain. What should I do?",
        "ನನ್ನ ಭತ್ತದ ಗದ್ದೆಯಲ್ಲಿ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ನಾನು ಏನು ಪರಿಶೀಲಿಸಬೇಕು?": "My rice paddy is raining too much. What should I check?",
        "ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳು ಕಾಣಿಸುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?": "I am seeing holes in my maize crop leaves. What should I do?",
        "ನನ್ನ ಕಡಲೆಕಾಯಿ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳು ಕಾಣಿಸುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?": "My groundnut crop leaves are showing spots. What should I do?",
        "ನನ್ನ ಹತ್ತಿ ಬೆಳೆಯಲ್ಲಿ ಕೀಟಗಳು ಕಾಣಿಸುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?": "There are insects attacking my cotton crop. What should I do?",
        "ನನ್ನ ಕಬ್ಬಿನ ಬೆಳೆಯ ಬೆಳವಣಿಗೆ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ. ಏನು ಮಾಡಬೇಕು?": "My sugarcane crop is not growing properly. What should I do?",
        "ರೈತರಿಗೆ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು ಯಾವುವು?": "What are the government schemes available for farmers?",
        "PM-KISAN ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.": "I need information about the PM-KISAN scheme.",
        "ನಾನು ರಾಗಿ ಬೆಳೆಸುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?": "I am growing ragi. Which government schemes may be applicable to me?",
        "ನನ್ನ ಬೆಳೆ ಹಾನಿಯಾಗಿದೆ. ಬೆಳೆ ವಿಮೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.": "My crop is damaged. I need information about crop insurance.",
        "XYZ ಕೃಷಿ ಯೋಜನೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ.": "Give information about XYZ agriculture scheme.",
        "ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?": "How to repair a laptop?"
    }

    SAMPLE_EN_TO_KN: Dict[str, str] = {
        "What are common causes of yellow leaves in tomato plants, and what should a farmer check first?": "ಟೊಮೇಟೊ ಗಿಡಗಳಲ್ಲಿ ಹಳದಿ ಎಲೆಗಳಿಗೆ ಸಾಮಾನ್ಯ ಕಾರಣಗಳು ಪೋಷಕಾಂಶಗಳ ಕೊರತೆ ಅಥವಾ ಅತಿಯಾದ ನೀರಾವರಿ. ಮೊದಲು ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ.",
        "There has been very little rain and my ragi crop is drying. What should I do?": "ರಾಗಿ ಬೆಳೆಗೆ ನೀರಿನ ಕೊರತೆಯಾದಾಗ ರಕ್ಷಣಾತ್ಮಕ ನೀರಾವರಿ ನೀಡಿ ಮತ್ತು ತೇವಾಂಶ ಸಂರಕ್ಷಣೆಗೆ ಮಣ್ಣಿನ ಹೊದಿಕೆ ಮಾಡಿ.",
        "Under PM-KISAN, eligible landholding farmer families receive ₹6,000 per year in 3 equal installments of ₹2,000 via DBT. Farmers must complete mandatory eKYC and land seeding on pmkisan.gov.in or through their local Raitha Samparka Kendra.": "ಪಿಎಂ-ಕಿಸಾನ್ ಯೋಜನೆಯಡಿ ಅರ್ಹ ರೈತ ಕುಟುಂಬಗಳಿಗೆ ವರ್ಷಕ್ಕೆ ₹6,000 (ತಲಾ ₹2,000 ರ ಮೂರು ಕಂತುಗಳಲ್ಲಿ) ಡಿಬಿಟಿ ಮೂಲಕ ನೇರವಾಗಿ ಜಮೆಯಾಗುತ್ತದೆ. ರೈತರು ಕಡ್ಡಾಯವಾಗಿ ಇ-ಕೆವೈಸಿ ಮತ್ತು ಭೂದಾಖಲೆ ಜೋಡಣೆ ಮಾಡಿಸಬೇಕು.",
        "Under Pradhan Mantri Fasal Bima Yojana (PMFBY) in Karnataka, farmers can insure notified crops through the Samrakshane portal. Premium is capped at 2% for Kharif food/oilseed crops and 1.5% for Rabi crops, with government subsidies covering the remainder.": "ಕರ್ನಾಟಕದಲ್ಲಿ ಪ್ರಧಾನ ಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆಯಡಿ ಸಂರಕ್ಷಣೆ ಪೋರ್ಟಲ್ ಮೂಲಕ ಬೆಳೆ ವಿಮೆ ಮಾಡಿಸಬಹುದು. ಖಾರೀಫ್ ಆಹಾರ/ಎಣ್ಣೆಕಾಳು ಬೆಳೆಗಳಿಗೆ ಶೇ 2 ಮತ್ತು ರಬಿ ಬೆಳೆಗಳಿಗೆ ಶೇ 1.5 ಪ್ರೀಮಿಯಂ ಇರುತ್ತದೆ."
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

    # Agricultural terminology alignment for Karnataka crops
    KN_CROP_CORRECTIONS = {
        "ರಾಗಿ": "ragi",
        "ಮೆಕ್ಕೆಜೋಳ": "maize",
        "ಕಡಲೆಕಾಯಿ": "groundnut",
        "ಕಬ್ಬು": "sugarcane",
        "ಹತ್ತಿ": "cotton",
        "ಮೆಣಸಿನಕಾಯಿ": "chilli",
        "ಈರುಳ್ಳಿ": "onion",
        "ಆಲೂಗಡ್ಡೆ": "potato",
        "ಬಾಳೆ": "banana",
        "ಟೊಮೇಟೊ": "tomato"
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

            translated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Align specific Kannada crop terms when translating kn -> en
            if src_lang == "kn" and tgt_lang == "en":
                if "ರಾಗಿ" in text:
                    # NLLB occasionally mistranslates 'ರಾಗಿ' as 'rice'
                    translated_text = re.sub(r"\brice\b", "ragi", translated_text, flags=re.IGNORECASE)
                if "ಮೆಕ್ಕೆಜೋಳ" in text:
                    translated_text = re.sub(r"\bcorn\b", "maize", translated_text, flags=re.IGNORECASE)

            return translated_text

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
