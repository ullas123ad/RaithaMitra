"""
Unit Tests for Canonical Crop Identity & RAG Grounding in RaithaMitra.

Validates:
1. Deterministic normalization for all 11 supported Karnataka crops (English & Kannada script).
2. Detection of Kannada crop entities from raw farmer queries.
3. Canonical crop precedence over NLLB translation variances.
4. RAG retrieval precision with strict cross-crop contamination rejection.
5. Regression test: Long Ragi query (Melukote) retrieves Ragi drought knowledge, NOT Chilli document.
"""

import unittest
from typing import Dict, Any

from model.advisory.crop_identifier import (
    SUPPORTED_CROPS,
    CROP_CANONICAL_MAP,
    normalize_crop_name,
    detect_crop_from_text,
    resolve_canonical_crop,
)
from model.advisory.retriever import AgriculturalRetriever
from model.advisory.language_bridge import MockLanguageBridge
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)


class TestCropIdentityAndNormalization(unittest.TestCase):
    """Tests canonical crop normalization and Kannada script detection."""

    def test_all_supported_crops_present(self):
        """Verify all 11 core Karnataka crops are in the supported registry."""
        expected = [
            "ragi", "paddy", "maize", "groundnut", "sugarcane",
            "cotton", "chilli", "onion", "potato", "banana", "tomato"
        ]
        for crop in expected:
            self.assertIn(crop, SUPPORTED_CROPS)
            self.assertEqual(normalize_crop_name(crop), crop)

    def test_kannada_chilli_detection(self):
        """Verify Kannada 'ಮೆಣಸಿನಕಾಯಿ' normalizes to 'chilli'."""
        self.assertEqual(normalize_crop_name("ಮೆಣಸಿನಕಾಯಿ"), "chilli")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ."), "chilli")

    def test_kannada_onion_detection(self):
        """Verify Kannada 'ಈರುಳ್ಳಿ' normalizes to 'onion'."""
        self.assertEqual(normalize_crop_name("ಈರುಳ್ಳಿ"), "onion")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಈರುಳ್ಳಿ ಬೆಳೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ."), "onion")

    def test_kannada_paddy_detection(self):
        """Verify Kannada 'ಭತ್ತ' normalizes to 'paddy'."""
        self.assertEqual(normalize_crop_name("ಭತ್ತ"), "paddy")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಭತ್ತದ ಗದ್ದೆಯಲ್ಲಿ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ."), "paddy")

    def test_kannada_ragi_detection(self):
        """Verify Kannada 'ರಾಗಿ' normalizes to 'ragi'."""
        self.assertEqual(normalize_crop_name("ರಾಗಿ"), "ragi")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ."), "ragi")

    def test_kannada_maize_detection(self):
        """Verify Kannada 'ಮೆಕ್ಕೆಜೋಳ' normalizes to 'maize'."""
        self.assertEqual(normalize_crop_name("ಮೆಕ್ಕೆಜೋಳ"), "maize")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳು ಕಾಣಿಸುತ್ತಿವೆ."), "maize")

    def test_kannada_groundnut_detection(self):
        """Verify Kannada 'ಕಡಲೆಕಾಯಿ' normalizes to 'groundnut'."""
        self.assertEqual(normalize_crop_name("ಕಡಲೆಕಾಯಿ"), "groundnut")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಕಡಲೆಕಾಯಿ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳು ಕಾಣಿಸುತ್ತಿವೆ."), "groundnut")

    def test_kannada_sugarcane_detection(self):
        """Verify Kannada 'ಕಬ್ಬು' normalizes to 'sugarcane'."""
        self.assertEqual(normalize_crop_name("ಕಬ್ಬು"), "sugarcane")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಕಬ್ಬಿನ ಬೆಳೆಯ ಬೆಳವಣಿಗೆ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ."), "sugarcane")

    def test_kannada_cotton_detection(self):
        """Verify Kannada 'ಹತ್ತಿ' normalizes to 'cotton'."""
        self.assertEqual(normalize_crop_name("ಹತ್ತಿ"), "cotton")
        self.assertEqual(detect_crop_from_text("ನನ್ನ ಹತ್ತಿ ಬೆಳೆಯಲ್ಲಿ ಕೀಟಗಳು ಕಾಣಿಸುತ್ತಿವೆ."), "cotton")

    def test_kannada_potato_detection(self):
        """Verify Kannada 'ಆಲೂಗಡ್ಡೆ' normalizes to 'potato'."""
        self.assertEqual(normalize_crop_name("ಆಲೂಗಡ್ಡೆ"), "potato")
        self.assertEqual(detect_crop_from_text("ಆಲೂಗಡ್ಡೆ ಬೆಳೆಗೆ ಬ್ಲೈಟ್ ರೋಗ ಬಂದಿದೆ."), "potato")

    def test_kannada_banana_detection(self):
        """Verify Kannada 'ಬಾಳೆ' normalizes to 'banana'."""
        self.assertEqual(normalize_crop_name("ಬಾಳೆ"), "banana")
        self.assertEqual(detect_crop_from_text("ಬಾಳೆ ಗಿಡದ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ."), "banana")

    def test_kannada_tomato_detection(self):
        """Verify Kannada 'ಟೊಮ್ಯಾಟೊ' normalizes to 'tomato'."""
        self.assertEqual(normalize_crop_name("ಟೊಮ್ಯಾಟೊ"), "tomato")
        self.assertEqual(detect_crop_from_text("ಟೊಮ್ಯಾಟೊ ಹಣ್ಣುಗಳಲ್ಲಿ ಕಲೆಗಳು ಕಾಣಿಸುತ್ತಿವೆ."), "tomato")

    def test_canonical_crop_priority_over_mistranslation(self):
        """Verify original Kannada query takes strict priority over translation variance."""
        # Case A: Kannada chilli ('ಮೆಣಸಿನಕಾಯಿ') translated as 'cucumber'
        resolved = resolve_canonical_crop(
            query="ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ.",
            translated_query="My cucumber leaves are falling. What should I do?"
        )
        self.assertEqual(resolved, "chilli")

        # Case B: Kannada onion ('ಈರುಳ್ಳಿ') translated as 'potato'
        resolved = resolve_canonical_crop(
            query="ನನ್ನ ಈರುಳ್ಳಿ ಬೆಳೆಗೆ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ.",
            translated_query="My potato crop is getting too much rain. What should I do?"
        )
        self.assertEqual(resolved, "onion")

        # Case C: Explicit crop argument takes highest priority
        resolved = resolve_canonical_crop(
            query="ನನ್ನ ಗದ್ದೆಯಲ್ಲಿ ಹೆಚ್ಚು ಮಳೆಯಾಗಿದೆ.",
            translated_query="My field is getting too much rain.",
            explicit_crop="paddy"
        )
        self.assertEqual(resolved, "paddy")


class TestRAGRetrieverCropGrounding(unittest.TestCase):
    """Tests RAG retriever with canonical crop grounding and strict cross-crop filtering."""

    def setUp(self):
        self.retriever = AgriculturalRetriever()

    def test_case_a_chilli_mistranslation_retrieval(self):
        """CASE A: Chilli query with mistranslated text retrieves chilli leaf curl knowledge."""
        docs = self.retriever.retrieve(
            query="My cucumber leaves are falling. What should I do?",
            crop="chilli"
        )
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0]["crop"], "chilli")
        self.assertEqual(docs[0]["id"], "chilli_pest_thrips_leaf_curl_013")

    def test_case_b_onion_mistranslation_retrieval(self):
        """CASE B: Onion query with mistranslated text retrieves onion excess rain knowledge."""
        docs = self.retriever.retrieve(
            query="My potato crop is getting too much rain. What should I do?",
            crop="onion"
        )
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0]["crop"], "onion")
        self.assertEqual(docs[0]["id"], "onion_excess_rainfall_purple_blotch_015")

    def test_case_c_paddy_excess_rain_retrieval(self):
        """CASE C: Paddy excess rain query retrieves general drainage or paddy context without cross-crop contamination."""
        docs = self.retriever.retrieve(
            query="My rice paddy is raining too much. What should I check?",
            crop="paddy"
        )
        # Should not retrieve maize, chilli, or onion
        for doc in docs:
            self.assertIn(doc["crop"], ["paddy", "general"])

    def test_case_d_ragi_drought_retrieval(self):
        """CASE D: Ragi drought query retrieves Ragi moisture stress knowledge."""
        docs = self.retriever.retrieve(
            query="My ragi crop is getting dry because it is not raining properly. What should I do?",
            crop="ragi"
        )
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0]["crop"], "ragi")
        self.assertEqual(docs[0]["id"], "ragi_drought_water_stress_003")

    def test_case_e_maize_pest_retrieval(self):
        """CASE E: Maize pest query retrieves Maize Fall Armyworm knowledge."""
        docs = self.retriever.retrieve(
            query="I have a few holes in my maize crop. What should I do?",
            crop="maize"
        )
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0]["crop"], "maize")
        self.assertEqual(docs[0]["id"], "maize_pest_fall_armyworm_005")

    def test_critical_regression_melukote_long_ragi_query(self):
        """
        CRITICAL REGRESSION TEST (Manual Test 21):
        Long Ragi query from Melukote MUST retrieve Ragi moisture stress knowledge,
        and MUST NOT retrieve chilli_pest_thrips_leaf_curl_013.
        """
        query_en = (
            "I am growing ragi in Melukote, Pandavapura taluk, Mandya district. "
            "For the past few days, it has not rained properly and the plants are looking dried up. "
            "The leaves are also curled slightly. Now what should I do?"
        )
        docs = self.retriever.retrieve(query=query_en, crop="ragi")

        self.assertTrue(len(docs) > 0, "Expected at least one retrieved document for Ragi query")
        self.assertEqual(docs[0]["id"], "ragi_drought_water_stress_003", "Rank 1 must be Ragi drought knowledge")
        self.assertEqual(docs[0]["crop"], "ragi")

        # Confirm chilli document is strictly excluded
        retrieved_ids = [d["id"] for d in docs]
        self.assertNotIn("chilli_pest_thrips_leaf_curl_013", retrieved_ids)

    def test_strict_cross_crop_rejection(self):
        """Verify that a target crop strictly rejects documents from other specific crops."""
        # Query mentioning leaves/rot on Ragi should never score Chilli or Cotton docs
        score_chilli = self.retriever.score_document(
            query="leaf curling and spots on leaves",
            doc={"crop": "chilli", "keywords": ["leaf curling", "thrips"], "title": "Chilli Leaf Curl", "content": "curling"},
            target_crop="ragi"
        )
        self.assertEqual(score_chilli, 0.0, "Score for different specific crop must be strictly 0.0")

        score_cotton = self.retriever.score_document(
            query="excess rainfall yellow leaves and waterlogging",
            doc={"crop": "cotton", "keywords": ["waterlogging", "yellowing"], "title": "Cotton Waterlogging", "content": "yellow leaves"},
            target_crop="maize"
        )
        self.assertEqual(score_cotton, 0.0, "Score for different specific crop must be strictly 0.0")


class TestAdvisoryEngineWithCropIdentity(unittest.TestCase):
    """Tests AdvisoryEngine end-to-end routing with canonical crop identity."""

    def setUp(self):
        config = AdvisoryConfig(backend="mock", use_rag=True)
        self.engine = AdvisoryEngine(
            config=config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            retriever=AgriculturalRetriever()
        )

    def test_advisory_engine_resolves_and_attaches_canonical_crop(self):
        """Verify AdvisoryEngine attaches canonical_crop to result schema."""
        result = self.engine.generate_advisory(
            query="ನನ್ನ ಮೆಣಸಿನಕಾಯಿ ಗಿಡದ ಎಲೆಗಳು ಮುದುರುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?",
            source_language="kn"
        )
        self.assertEqual(result["canonical_crop"], "chilli")
        self.assertTrue(result["rag_enabled"])
        self.assertTrue(len(result["retrieved_documents"]) > 0)
        self.assertEqual(result["retrieved_documents"][0]["crop"], "chilli")


if __name__ == "__main__":
    unittest.main()
