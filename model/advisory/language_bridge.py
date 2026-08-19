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
        "ನನಗೆ PM-KISAN ಹಣದ ಕಂತು ಬಂದಿಲ್ಲ. ನಾನು ಏನು ಮಾಡಬೇಕು?": "I have not received my PM-KISAN installment. What should I do?",
        "ನಾನು ರಾಗಿ ಬೆಳೆಸುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?": "I am growing ragi. Which government schemes may be applicable to me?",
        "ನಾನು ರಾಗಿ ಬೆಳೆಯುತ್ತಿದ್ದೇನೆ. ನನಗೆ ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಸಂಬಂಧಿಸಬಹುದು?": "I am growing ragi. Which government schemes may be applicable to me?",
        "ನಾನು ರಾಗಿ ಬೆಳೆಸುತ್ತೇನೆ. ನನ್ನ ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು ಯಾವುವು?": "I am cultivating ragi. What government schemes are relevant to my crop?",
        "ನನ್ನ ಬೆಳೆ ಹಾನಿಯಾಗಿದೆ. ಬೆಳೆ ವಿಮೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.": "My crop is damaged. I need information about crop insurance.",
        "ನನ್ನ ರಾಗಿ ಬೆಳೆ ಮಳೆಯಿಂದ ಹಾನಿಯಾಗಿದೆ. ಬೆಳೆ ವಿಮೆಯಿಂದ ಸಹಾಯ ಸಿಗುತ್ತದೆಯೇ?": "My ragi crop is damaged due to rain. Can I get help from crop insurance?",
        "ನನ್ನ ಹೊಲಕ್ಕೆ ಡ್ರಿಪ್ ನೀರಾವರಿ ಹಾಕಿಸಿಕೊಳ್ಳಲು ಸರ್ಕಾರದಿಂದ ಸಬ್ಸಿಡಿ ಇದೆಯೇ?": "Is there a government subsidy to install drip irrigation in my field?",
        "ಕೃಷಿ ಯಂತ್ರೋಪಕರಣಗಳನ್ನು ಖರೀದಿಸಲು ಸರ್ಕಾರದಿಂದ ಸಬ್ಸಿಡಿ ಸಿಗುತ್ತದೆಯೇ?": "Is there a subsidy from the government to purchase farm machinery?",
        "ಕರ್ನಾಟಕ ಸರ್ಕಾರದ XYZ ರೈತ ಯೋಜನೆಗೆ ನಾನು ಹೇಗೆ ಅರ್ಜಿ ಹಾಕಬೇಕು?": "How do I apply for the Karnataka Government XYZ farmer scheme?",
        "XYZ ಕೃಷಿ ಯೋಜನೆ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ.": "Give information about XYZ agriculture scheme.",
        "ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?": "How to repair a laptop?",
        "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಯಾವುದು ಹಾಕಬೇಕು?": "What fertilizer should I apply for my ragi crop?",
        "ನನ್ನ ಹೊಲದ ಮಣ್ಣಿನಲ್ಲಿ ಪೋಷಕಾಂಶ ಕಡಿಮೆಯಾಗಿದೆ ಎಂದು ತಿಳಿಯುವುದು ಹೇಗೆ?": "How to know if nutrients are deficient in my farm's soil?",
        "ನನ್ನ ಭತ್ತದ ಗದ್ದೆಯ ಮಣ್ಣಿನ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.": "I need information about the soil in my paddy field.",
        "ನನ್ನ ಹೊಲದ ಮಣ್ಣಿನಲ್ಲಿ ಪೋಷಕಾಂಶ ಕಡಿಮೆಯಿದೆಯೇ?": "Are nutrients deficient in my farm's soil?",
        "ನನ್ನ ಮಣ್ಣಿಗೆ ಯಾವ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?": "Which fertilizer should I apply to my soil?",
        "ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ.": "My laptop is not working properly.",
        "ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?": "What is the ragi price in Mandya?",
        "ಬೆಳಗಾವಿಯಲ್ಲಿ ಮೆಕ್ಕೆಜೋಳದ ಇತ್ತೀಚಿನ ಬೆಲೆ ಎಷ್ಟು?": "What is the latest price of maize in Belagavi?",
        "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಟೊಮ್ಯಾಟೊ ಬೆಲೆ ಎಷ್ಟು?": "What is the tomato price in Bengaluru?",
        "ಈರುಳ್ಳಿಗೆ ಕರ್ನಾಟಕದಲ್ಲಿ ಯಾವ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಉತ್ತಮ ಬೆಲೆ ಇದೆ?": "Which market in Karnataka has a better price for onion?",
        "ಇವತ್ತು ರಾಗಿಯ ಬೆಲೆ ಎಷ್ಟು?": "What is the ragi price today?",
        "ನನ್ನ ಮೆಕ್ಕೆಜೋಳವನ್ನು ಯಾವ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಮಾರಬಹುದು?": "In which market can I sell my maize?",
        "ಮಂಡ್ಯದಲ್ಲಿ ಟೊಮ್ಯಾಟೊ ಬೆಲೆ ಎಷ್ಟು?": "What is the tomato price in Mandya?",
        "ನನ್ನ ಈರುಳ್ಳಿಗೆ ಯಾವ APMC ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಉತ್ತಮ ಬೆಲೆ ಇದೆ?": "Which APMC market has the best price for my onion?",
        "ಕಳೆದ ಕೆಲವು ದಿನಗಳಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಹೇಗಿದೆ?": "How is the ragi price in recent days?",
        "ಇವತ್ತು ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಎಷ್ಟು?": "What is today's market price?",
        "ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಬೆಲೆ ಎಷ್ಟು?": "What is the price of my laptop?",
        "ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?": "How to repair a laptop?",
        "ನಾನು ಕೇಸರಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೇನೆ.": "I want to grow saffron.",
        "ಉಡುಪಿಯಲ್ಲಿ ನನ್ನ ಭತ್ತದ ಗದ್ದೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ. ಏನು ಪರಿಶೀಲಿಸಬೇಕು?": "My paddy field in Udupi has received heavy rainfall. What should I check?",
        "ಬೆಳಗಾವಿಯಲ್ಲಿ ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳಿವೆ. ಏನು ಮಾಡಬೇಕು?": "My maize crop leaves in Belagavi have holes. What should I do?",
        "ರಾಗಿ": "ragi",
        "ಭತ್ತ": "paddy",
        "ಮೆಕ್ಕೆಜೋಳ": "maize",
        "My ragi price in Mandya today": "My ragi price in Mandya today"
    }

    SAMPLE_EN_TO_KN: Dict[str, str] = {
        "What are common causes of yellow leaves in tomato plants, and what should a farmer check first?": "ಟೊಮೇಟೊ ಗಿಡಗಳಲ್ಲಿ ಹಳದಿ ಎಲೆಗಳಿಗೆ ಸಾಮಾನ್ಯ ಕಾರಣಗಳು ಪೋಷಕಾಂಶಗಳ ಕೊರತೆ ಅಥವಾ ಅತಿಯಾದ ನೀರಾವರಿ. ಮೊದಲು ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ.",
        "There has been very little rain and my ragi crop is drying. What should I do?": "ರಾಗಿ ಬೆಳೆಗೆ ನೀರಿನ ಕೊರತೆಯಾದಾಗ ರಕ್ಷಣಾತ್ಮಕ ನೀರಾವರಿ ನೀಡಿ ಮತ್ತು ತೇವಾಂಶ ಸಂರಕ್ಷಣೆಗೆ ಮಣ್ಣಿನ ಹೊದಿಕೆ ಮಾಡಿ.",
        "Under PM-KISAN, eligible landholding farmer families receive ₹6,000 per year in 3 equal installments of ₹2,000 via DBT. Farmers must complete mandatory eKYC and land seeding on pmkisan.gov.in or through their local Raitha Samparka Kendra.": "ಪಿಎಂ-ಕಿಸಾನ್ ಯೋಜನೆಯಡಿ ಅರ್ಹ ರೈತ ಕುಟುಂಬಗಳಿಗೆ ವರ್ಷಕ್ಕೆ ₹6,000 (ತಲಾ ₹2,000 ರ ಮೂರು ಕಂತುಗಳಲ್ಲಿ) ಡಿಬಿಟಿ ಮೂಲಕ ನೇರವಾಗಿ ಜಮೆಯಾಗುತ್ತದೆ. ರೈತರು ಕಡ್ಡಾಯವಾಗಿ ಇ-ಕೆವೈಸಿ ಮತ್ತು ಭೂದಾಖಲೆ ಜೋಡಣೆ ಮಾಡಿಸಬೇಕು.",
        "Under Pradhan Mantri Fasal Bima Yojana (PMFBY) in Karnataka, farmers can insure notified crops through the Samrakshane portal. Premium is capped at 2% for Kharif food/oilseed crops and 1.5% for Rabi crops, with government subsidies covering the remainder.": "ಕರ್ನಾಟಕದಲ್ಲಿ ಪ್ರಧಾನ ಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆಯಡಿ ಸಂರಕ್ಷಣೆ ಪೋರ್ಟಲ್ ಮೂಲಕ ಬೆಳೆ ವಿಮೆ ಮಾಡಿಸಬಹುದು. ಖಾರೀಫ್ ಆಹಾರ/ಎಣ್ಣೆಕಾಳು ಬೆಳೆಗಳಿಗೆ ಶೇ 2 ಮತ್ತು ರಬಿ ಬೆಳೆಗಳಿಗೆ ಶೇ 1.5 ಪ್ರೀಮಿಯಂ ಇರುತ್ತದೆ.",
        "Under PMKSY Per Drop More Crop, assistance is available for micro-irrigation (drip/sprinkler). In Karnataka, combined subsidies reach up to 90% for SC/ST and 75% for general category farmers subject to official verification.": "ಪ್ರಧಾನ ಮಂತ್ರಿ ಕೃಷಿ ಸಿಂಚಾಯಿ ಯೋಜನೆಯಡಿ (ಡ್ರಾಪ್ ಮೋರ್ ಕ್ರಾಪ್) ಹನಿ ಮತ್ತು ತುಂತುರು ನೀರಾವರಿಗೆ ಸಹಾಯಧನ ಲಭ್ಯವಿದೆ. ಕರ್ನಾಟಕದಲ್ಲಿ ಎಸ್‌ಸಿ/ಎಸ್‌ಟಿ ರೈತರಿಗೆ ಶೇ 90 ರವರೆಗೆ ಮತ್ತು ಸಾಮಾನ್ಯ ವರ್ಗದ ರೈತರಿಗೆ ಶೇ 75 ರವರೆಗೆ ಸಹಾಯಧನ ಲಭ್ಯವಿರುತ್ತದೆ.",
        "Under Sub-Mission on Agricultural Mechanization (SMAM) in Karnataka, subsidies of 40% to 50% for general farmers and 50% to 90% for SC/ST farmers are available for approved farm equipment subject to department targets.": "ಕರ್ನಾಟಕದಲ್ಲಿ ಕೃಷಿ ಯಾಂತ್ರೀಕರಣ ಯೋಜನೆಯಡಿ (SMAM) ಸಾಮಾನ್ಯ ರೈತರಿಗೆ ಶೇ 40-50 ಮತ್ತು ಎಸ್‌ಸಿ/ಎಸ್‌ಟಿ ರೈತರಿಗೆ ಶೇ 50-90 ರವರೆಗೆ ಅನುಮೋದಿತ ಯಂತ್ರೋಪಕರಣಗಳಿಗೆ ಸಹಾಯಧನ ಲಭ್ಯವಿದೆ.",
        "No verified government scheme was found matching this name. Please verify official schemes at your local Raitha Samparka Kendra or on the official Karnataka Agriculture portal (raitamitra.karnataka.gov.in).": "ಈ ಹೆಸರಿನ ಯಾವುದೇ ಅಧಿಕೃತ ಸರ್ಕಾರಿ ಯೋಜನೆ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರ ಅಥವಾ ಅಧಿಕೃತ ಕೃಷಿ ಪೋರ್ಟಲ್ (raitamitra.karnataka.gov.in) ನಲ್ಲಿ ಪರಿಶೀಲಿಸಿ.",
        "For balanced fertilizer management, do not apply arbitrary chemical dosages. Regional soils benefit from organic Farm Yard Manure (FYM). For crop-specific NPK dosages, obtain an official Soil Health Card test from your local Raitha Samparka Kendra.": "ಸಮತೋಲಿತ ಗೊಬ್ಬರ ನಿರ್ವಹಣೆಗೆ ಅಂದಾಜಿನ ಮೇಲೆ ರಾಸಾಯನಿಕ ಗೊಬ್ಬರ ಬಳಸಬೇಡಿ. ಸಾವಯವ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ಬಳಸಿ ಮತ್ತು ನಿಖರ ಪೋಷಕಾಂಶಗಳಿಗಾಗಿ ಸ್ಥಳೀಯ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರದಲ್ಲಿ ಮಣ್ಣು ಆರೋಗ್ಯ ಪತ್ರಿಕೆ (Soil Health Card) ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ.",
        "Regional soil health information provides general soil texture and classification. For field-specific nutrient diagnosis and Soil Health Card testing, visit your local Raitha Samparka Kendra (RSK).": "ಪ್ರಾದೇಶಿಕ ಮಣ್ಣಿನ ಮಾಹಿತಿಯು ಸಾಮಾನ್ಯ ಮಣ್ಣಿನ ವಿಧವನ್ನು ತಿಳಿಸುತ್ತದೆ. ನಿಮ್ಮ ಹೊಲದ ನಿಖರ ಪೋಷಕಾಂಶ ಪರೀಕ್ಷೆಗಾಗಿ ಸ್ಥಳೀಯ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರದಲ್ಲಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ.",
        "At Mandya APMC on 2026-08-19, reported Ragi prices ranged from ₹2,800 to ₹3,400 per quintal with a modal price of ₹3,200/quintal (Arrivals: 45 tonnes). Note that actual price received depends on grain quality and moisture.": "2026-08-19 ರಂದು ಮಂಡ್ಯ ಎಪಿಎಂಸಿಯಲ್ಲಿ ವರದಿಯಾದ ರಾಗಿ ಬೆಲೆ ಕ್ವಿಂಟಾಲ್‌ಗೆ ₹2,800 ರಿಂದ ₹3,400 ಇದ್ದು, ಮಾದರಿ ಬೆಲೆ ₹3,200/ಕ್ವಿಂಟಾಲ್ ಆಗಿದೆ (ಆವಕ: 45 ಟನ್).",
        "At Belagavi APMC, latest available reported Maize price is dated 2026-08-18 with a modal price of ₹2,350/quintal (Range: ₹2,100 - ₹2,450/quintal, Arrivals: 120 tonnes).": "ಬೆಳಗಾವಿ ಎಪಿಎಂಸಿಯಲ್ಲಿ ಲಭ್ಯವಿರುವ ಇತ್ತೀಚಿನ ಮೆಕ್ಕೆಜೋಳದ ಬೆಲೆ 2026-08-18 ರ ವರದಿಯಂತೆ ಮಾದರಿ ಬೆಲೆ ₹2,350/ಕ್ವಿಂಟಾಲ್ ಆಗಿದೆ (ದರ ಶ್ರೇಣಿ: ₹2,100 - ₹2,450/ಕ್ವಿಂಟಾಲ್, ಆವಕ: 120 ಟನ್).",
        "At Binny Mill (F&V) APMC in Bengaluru on 2026-08-19, reported Tomato prices ranged from ₹1,400 to ₹2,200 per quintal with a modal price of ₹1,800/quintal (Arrivals: 250 tonnes).": "2026-08-19 ರಂದು ಬೆಂಗಳೂರಿನ ಬಿನ್ನಿಮಿಲ್ ಎಪಿಎಂಸಿಯಲ್ಲಿ ಟೊಮ್ಯಾಟೊ ಬೆಲೆ ಕ್ವಿಂಟಾಲ್‌ಗೆ ₹1,400 ರಿಂದ ₹2,200 ಇದ್ದು, ಮಾದರಿ ಬೆಲೆ ₹1,800/ಕ್ವಿಂಟಾಲ್ ಆಗಿದೆ (ಆವಕ: 250 ಟನ್).",
        "Based on reported official APMC data for 2026-08-19, Yeshwanthpur Bengaluru reported a modal price of ₹2,500/quintal (Range: ₹2,000 - ₹2,800) and Hubballi APMC reported ₹2,300/quintal (Range: ₹1,800 - ₹2,600). Confirm current auction rates before transporting produce.": "2026-08-19 ರ ಎಪಿಎಂಸಿ ವರದಿಯ ಪ್ರಕಾರ, ಯಶವಂತಪುರ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಈರುಳ್ಳಿ ಮಾದರಿ ಬೆಲೆ ₹2,500/ಕ್ವಿಂಟಾಲ್ (ಶ್ರೇಣಿ: ₹2,000 - ₹2,800) ಮತ್ತು ಹುಬ್ಬಳ್ಳಿ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ₹2,300/ಕ್ವಿಂಟಾಲ್ (ಶ್ರೇಣಿ: ₹1,800 - ₹2,600) ಆಗಿದೆ.",
        "Official APMC market prices vary by district and daily trading session. Please specify your crop and nearby APMC market to check latest reported prices.": "ಅಧಿಕೃತ ಎಪಿಎಂಸಿ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು ಜಿಲ್ಲೆ ಮತ್ತು ದಿನದ ವಹಿವಾಟಿಗೆ ಅನುಗುಣವಾಗಿ ಬದಲಾಗುತ್ತವೆ. ಇತ್ತೀಚಿನ ಬೆಲೆ ತಿಳಿಯಲು ನಿಮ್ಮ ಬೆಳೆ ಮತ್ತು ಸಮೀಪದ ಮಾರುಕಟ್ಟೆಯನ್ನು ನಿರ್ದಿಷ್ಟಪಡಿಸಿ.",
        "RaithaMitra is an agricultural advisory assistant dedicated to crop health, weather, soil, farming schemes, and market prices. Please ask an agriculture-related question.": "ರೈತಮಿತ್ರ (RaithaMitra) ಕೃಷಿ ಸಲಹಾ ವ್ಯವಸ್ಥೆಯಾಗಿದ್ದು, ಬೆಳೆಗಳ ಆರೋಗ್ಯ, ಹವಾಮಾನ, ಮಣ್ಣು, ಕೃಷಿ ಯೋಜನೆಗಳು ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲದು. ದಯವಿಟ್ಟು ಕೃಷಿ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.",
        "Specific local package of practices for this crop is not currently available in the Karnataka agricultural knowledge base. Please consult your nearest Krishi Vigyan Kendra (KVK) or University of Agricultural Sciences for specialized guidance.": "ಈ ಬೆಳೆಯ ಕುರಿತು ಕರ್ನಾಟಕ ಕೃಷಿ ಮಾಹಿತಿ ಕೋಶದಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ಸ್ಥಳೀಯ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ಕೃಷಿ ವಿಶ್ವವಿದ್ಯಾಲಯವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "What issue are you facing with your crop? Please specify if you need guidance regarding leaf symptoms, pests, diseases, irrigation, fertilizer/soil, market prices, or government schemes.": "ನಿಮ್ಮ ಬೆಳೆಯಲ್ಲಿ ನಿಮಗೆ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ? ಎಲೆಗಳ ಲಕ್ಷಣ, ಕೀಟ, ರೋಗ, ನೀರಾವರಿ, ಗೊಬ್ಬರ/ಮಣ್ಣು, ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಅಥವಾ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕೇ ಎಂದು ದಯವಿಟ್ಟು ತಿಳಿಸಿ.",
        "Excess rainfall and standing water cause root suffocation. Dig drainage trenches immediately and apply 1% urea foliar spray once stagnant water is cleared.": "ಅತಿಯಾದ ಮಳೆ ಮತ್ತು ನಿಂತ ನೀರು ಬೇರುಗಳ ಉಸಿರುಗಟ್ಟುವಿಕೆಗೆ ಕಾರಣವಾಗುತ್ತದೆ. ತಕ್ಷಣವೇ ಬಸಿಗಾಲುವೆಗಳನ್ನು ನಿರ್ಮಿಸಿ ನೀರನ್ನು ಹೊರಹಾಕಿ ಮತ್ತು ಶಿಲೀಂಧ್ರ ರೋಗಗಳ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "Maize fall armyworm requires prompt action: install pheromone traps @ 4/acre and apply Emamectin benzoate 5% SG (0.4g/L) inside the central whorl if holes appear.": "ಮೆಕ್ಕೆಜೋಳದ ಸೈನಿಕ ಹುಳು ಬಾಧೆಗೆ ಎಕರೆಗೆ 4 ಮೋಹಕ ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ. ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳಿದ್ದರೆ ಸುಳಿಯ ಒಳಗೆ ಎಮಾಮೆಕ್ಟಿನ್ ಬೆಂಜೊಯೇಟ್ 5% SG (0.4 ಗ್ರಾಂ/ಲೀಟರ್) ಸಿಂಪಡಿಸಿ.",
        "Key agricultural schemes available in Karnataka include PM-KISAN (direct income support), PMFBY (crop insurance via Samrakshane), Krishi Bhagya (farm pond and water conservation subsidy), and KCC (concessional crop loans). Farmers can verify eligibility via the FRUITS portal.": "ಕರ್ನಾಟಕದಲ್ಲಿ ರೈತರಿಗೆ ಲಭ್ಯವಿರುವ ಪ್ರಮುಖ ಯೋಜನೆಗಳೆಂದರೆ ಪಿಎಂ-ಕಿಸಾನ್, ಪಿಎಂಎಫ್‌ಬಿವೈ (ಸಂರಕ್ಷಣೆ ಪೋರ್ಟಲ್ ಮೂಲಕ ಬೆಳೆ ವಿಮೆ), ಕೃಷಿ ಭಾಗ್ಯ ಮತ್ತು ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್. ರೈತರು ಫ್ರೂಟ್ಸ್ (FRUITS) ಪೋರ್ಟಲ್ ಮೂಲಕ ತಮ್ಮ ಅರ್ಹತೆ ಪರಿಶೀಲಿಸಬಹುದು."
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
