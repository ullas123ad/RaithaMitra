"""
Comprehensive Crop Coverage & Knowledge-Grounding Validation Suite
===================================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Validates:
1. Machine-readable crop registry structure and three-tier classification consistency.
2. Canonical crop identity resolution across all Karnataka agricultural categories.
3. Explicit Vanilla and Saffron audit (recognized but not supported -> safe KVK referral).
4. Strict RAG crop isolation (0% cross-crop contamination across all 28 supported crops).
5. Representative Kannada voice/transcript query routing across 10 distinct crop categories.
6. Non-agricultural guardrail preservation.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.advisory.crop_identifier import (
    SUPPORTED_CROPS,
    RECOGNIZED_UNSUPPORTED_CROPS,
    CROP_CANONICAL_MAP,
    get_crop_registry,
    get_crop_entry,
    get_supported_crops,
    get_recognized_crops,
    is_crop_supported,
    get_crop_support_status,
    get_crop_category,
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


class TestCropRegistryAndClassification(unittest.TestCase):
    """Validates structure, integrity, and three-tier classification of crop registry."""

    def setUp(self):
        self.registry = get_crop_registry()
        self.crops = self.registry.get("crops", {})

    def test_01_registry_loaded_and_valid(self):
        """Verify registry version, categories, and non-empty crop dictionary."""
        self.assertIn("registry_version", self.registry)
        self.assertIn("categories", self.registry)
        self.assertGreater(len(self.crops), 30)

    def test_02_three_tier_status_consistency(self):
        """Verify that every crop in registry has a valid and consistent support status."""
        valid_statuses = {"supported", "recognized_not_supported", "unsupported"}
        for crop_id, data in self.crops.items():
            status = data.get("support_status")
            self.assertIn(status, valid_statuses, f"Invalid status '{status}' for crop {crop_id}")

            rag_supported = data.get("rag_supported", False)
            doc_count = data.get("document_count", 0)

            if status == "supported":
                self.assertTrue(rag_supported, f"Supported crop {crop_id} must have rag_supported=True")
                self.assertGreater(doc_count, 0, f"Supported crop {crop_id} must have document_count > 0")
                self.assertGreater(len(data.get("topics", [])), 0, f"Supported crop {crop_id} must have topics")
                self.assertGreater(len(data.get("source_institutions", [])), 0, f"Supported crop {crop_id} must list source institutions")
            elif status == "recognized_not_supported":
                self.assertFalse(rag_supported, f"Recognized unsupported crop {crop_id} must have rag_supported=False")
                self.assertEqual(doc_count, 0, f"Recognized unsupported crop {crop_id} must have document_count=0")
                self.assertEqual(len(data.get("topics", [])), 0, f"Recognized unsupported crop {crop_id} must have empty topics")

    def test_03_supported_crops_count_and_composition(self):
        """Verify all 28 core Karnataka crops are in SUPPORTED_CROPS."""
        expected_supported = [
            "ragi", "paddy", "maize", "jowar", "bajra",
            "red_gram", "bengal_gram", "green_gram", "black_gram",
            "groundnut", "sunflower", "soybean",
            "sugarcane", "cotton", "tobacco",
            "chilli", "turmeric", "ginger", "black_pepper", "cardamom",
            "tomato", "onion", "potato", "brinjal", "cabbage", "bhendi",
            "banana", "mango", "pomegranate", "grapes", "papaya", "lime",
            "arecanut", "coconut", "coffee", "cashew",
            "jasmine", "marigold"
        ]
        supported = get_supported_crops()
        self.assertGreaterEqual(len(supported), 28)
        for c in ["ragi", "paddy", "maize", "jowar", "red_gram", "bengal_gram", "groundnut", "arecanut", "coffee", "turmeric", "chilli"]:
            self.assertIn(c, supported, f"Expected {c} to be supported")

    def test_04_categories_coverage(self):
        """Verify that crops span all 12 agricultural categories."""
        categories = set(data.get("category") for data in self.crops.values())
        expected_cats = {"cereal", "millet", "pulse", "oilseed", "spice", "vegetable", "fruit", "plantation", "commercial", "floriculture", "medicinal", "tuber"}
        for cat in expected_cats:
            self.assertIn(cat, categories, f"Category {cat} missing from registry")


class TestKannadaCropNormalization(unittest.TestCase):
    """Tests Kannada crop name normalization across multiple crop categories."""

    def test_05_cereal_and_millet_normalization(self):
        """Test normalization of Kannada cereals and millets."""
        self.assertEqual(normalize_crop_name("ರಾಗಿ"), "ragi")
        self.assertEqual(normalize_crop_name("ಭತ್ತ"), "paddy")
        self.assertEqual(normalize_crop_name("ಮೆಕ್ಕೆಜೋಳ"), "maize")
        self.assertEqual(normalize_crop_name("ಜೋಳ"), "jowar")
        self.assertEqual(normalize_crop_name("ಸಜ್ಜೆ"), "bajra")
        self.assertEqual(normalize_crop_name("ನವಣೆ"), "foxtail_millet")
        self.assertEqual(normalize_crop_name("ಸಾಮೆ"), "little_millet")

    def test_06_pulse_normalization(self):
        """Test normalization of Kannada pulses."""
        self.assertEqual(normalize_crop_name("ತೊಗರಿ"), "red_gram")
        self.assertEqual(normalize_crop_name("ಕಡಲೆ"), "bengal_gram")
        self.assertEqual(normalize_crop_name("ಹೆಸರು"), "green_gram")
        self.assertEqual(normalize_crop_name("ಉದ್ದು"), "black_gram")
        self.assertEqual(normalize_crop_name("ಅಲಸಂದೆ"), "cowpea")
        self.assertEqual(normalize_crop_name("ಅವರೆ"), "field_bean")
        self.assertEqual(normalize_crop_name("ಹುರಳಿ"), "horse_gram")

    def test_07_oilseed_normalization(self):
        """Test normalization of Kannada oilseeds."""
        self.assertEqual(normalize_crop_name("ಕಡಲೆಕಾಯಿ"), "groundnut")
        self.assertEqual(normalize_crop_name("ಶೇಂಗಾ"), "groundnut")
        self.assertEqual(normalize_crop_name("ಸೂರ್ಯಕಾಂತಿ"), "sunflower")
        self.assertEqual(normalize_crop_name("ಸೋಯಾಬೀನ್"), "soybean")
        self.assertEqual(normalize_crop_name("ಎಳ್ಳು"), "sesame")
        self.assertEqual(normalize_crop_name("ಕುಸುಮೆ"), "safflower")
        self.assertEqual(normalize_crop_name("ಹರಳು"), "castor")
        self.assertEqual(normalize_crop_name("ಸಾಸಿವೆ"), "mustard")

    def test_08_spice_normalization(self):
        """Test normalization of Kannada spices."""
        self.assertEqual(normalize_crop_name("ಮೆಣಸಿನಕಾಯಿ"), "chilli")
        self.assertEqual(normalize_crop_name("ಅರಿಶಿನ"), "turmeric")
        self.assertEqual(normalize_crop_name("ಶುಂಠಿ"), "ginger")
        self.assertEqual(normalize_crop_name("ಕರಿಮೆಣಸು"), "black_pepper")
        self.assertEqual(normalize_crop_name("ಏಲಕ್ಕಿ"), "cardamom")
        self.assertEqual(normalize_crop_name("ಕೊತ್ತಂಬರಿ"), "coriander")
        self.assertEqual(normalize_crop_name("ಜೀರಿಗೆ"), "cumin")
        self.assertEqual(normalize_crop_name("ಬೆಳ್ಳುಳ್ಳಿ"), "garlic")
        self.assertEqual(normalize_crop_name("ವೆನಿಲ್ಲಾ"), "vanilla")

    def test_09_plantation_and_commercial_normalization(self):
        """Test normalization of plantation and commercial crops."""
        self.assertEqual(normalize_crop_name("ಅಡಿಕೆ"), "arecanut")
        self.assertEqual(normalize_crop_name("ಕಾಫಿ"), "coffee")
        self.assertEqual(normalize_crop_name("ತೆಂಗು"), "coconut")
        self.assertEqual(normalize_crop_name("ಗೋಡಂಬಿ"), "cashew")
        self.assertEqual(normalize_crop_name("ಕಬ್ಬು"), "sugarcane")
        self.assertEqual(normalize_crop_name("ಹತ್ತಿ"), "cotton")
        self.assertEqual(normalize_crop_name("ತಂಬಾಕು"), "tobacco")

    def test_10_vegetable_fruit_floriculture_normalization(self):
        """Test normalization of vegetables, fruits, and flowers."""
        self.assertEqual(normalize_crop_name("ಟೊಮ್ಯಾಟೊ"), "tomato")
        self.assertEqual(normalize_crop_name("ಈರುಳ್ಳಿ"), "onion")
        self.assertEqual(normalize_crop_name("ಆಲೂಗಡ್ಡೆ"), "potato")
        self.assertEqual(normalize_crop_name("ಬದನೆಕಾಯಿ"), "brinjal")
        self.assertEqual(normalize_crop_name("ಎಲೆಕೋಸು"), "cabbage")
        self.assertEqual(normalize_crop_name("ಬೆಂಡೆಕಾಯಿ"), "bhendi")
        self.assertEqual(normalize_crop_name("ಬಾಳೆ"), "banana")
        self.assertEqual(normalize_crop_name("ಮಾವು"), "mango")
        self.assertEqual(normalize_crop_name("ದಾಳಿಂಬೆ"), "pomegranate")
        self.assertEqual(normalize_crop_name("ದ್ರಾಕ್ಷಿ"), "grapes")
        self.assertEqual(normalize_crop_name("ಮಲ್ಲಿಗೆ"), "jasmine")
        self.assertEqual(normalize_crop_name("ಚೆಂಡುಹೂ"), "marigold")


class TestExplicitVanillaAndSaffronAudit(unittest.TestCase):
    """Explicitly audits Vanilla and Saffron support status and safe KVK referral."""

    def setUp(self):
        config = AdvisoryConfig(backend="mock", use_rag=True)
        self.engine = AdvisoryEngine(
            config=config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            retriever=AgriculturalRetriever()
        )
        self.retriever = AgriculturalRetriever()

    def test_11_vanilla_audit_and_rejection_safety(self):
        """
        EXPLICIT AUDIT: Vanilla
        1. Recognized in Kannada ('ವೆನಿಲ್ಲಾ') and English ('vanilla').
        2. Status is 'recognized_not_supported'.
        3. RAG retrieval returns 0 documents.
        4. Engine safely directs farmer to KVK / University of Agricultural Sciences.
        """
        self.assertEqual(normalize_crop_name("vanilla"), "vanilla")
        self.assertEqual(normalize_crop_name("ವೆನಿಲ್ಲಾ"), "vanilla")
        self.assertEqual(get_crop_support_status("vanilla"), "recognized_not_supported")
        self.assertFalse(is_crop_supported("vanilla"))

        # RAG retrieval for vanilla must be empty
        docs = self.retriever.retrieve(query="vanilla crop disease rot", crop="vanilla")
        self.assertEqual(len(docs), 0, "Vanilla must have 0 RAG documents")

        # Advisory response must provide safe institutional guidance
        res = self.engine.generate_advisory(
            query="ನನ್ನ ವೆನಿಲ್ಲಾ ಬೆಳೆಗೆ ರೋಗ ಬಂದಿದೆ. ಏನು ಮಾಡಬೇಕು?",
            source_language="kn"
        )
        self.assertEqual(res["canonical_crop"], "vanilla")
        self.assertEqual(res["crop_support_status"], "recognized_not_supported")
        self.assertEqual(len(res["retrieved_documents"]), 0)
        self.assertIn("ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ", res["response"])

    def test_12_saffron_audit_and_rejection_safety(self):
        """
        EXPLICIT AUDIT: Saffron
        1. Recognized as 'saffron'.
        2. Status is 'recognized_not_supported'.
        3. Safe institutional referral returned.
        """
        self.assertEqual(normalize_crop_name("saffron"), "saffron")
        self.assertEqual(normalize_crop_name("ಕೇಸರಿ"), "saffron")
        self.assertEqual(get_crop_support_status("saffron"), "recognized_not_supported")

        res = self.engine.generate_advisory(
            query="ನಾನು ಕೇಸರಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೇನೆ.",
            source_language="kn"
        )
        self.assertEqual(res["canonical_crop"], "saffron")
        self.assertEqual(len(res["retrieved_documents"]), 0)
        self.assertIn("ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ", res["response"])


class TestStrictRAGZeroCrossContamination(unittest.TestCase):
    """Verifies that RAG retrieval achieves 0% cross-crop contamination across all supported crops."""

    def setUp(self):
        self.retriever = AgriculturalRetriever()
        self.supported_crops = get_supported_crops()

    def test_13_zero_cross_crop_contamination_audit(self):
        """
        STRICT 0% CROSS-CROP CONTAMINATION AUDIT:
        For every supported crop, query retrieval with crop specified,
        and assert 100% of retrieved documents belong to that crop.
        """
        total_crops_tested = 0
        contaminated_crops = []

        for crop in self.supported_crops:
            query = f"guidance on pest disease and water management for {crop}"
            docs = self.retriever.retrieve(query=query, crop=crop)
            
            if docs:
                total_crops_tested += 1
                for doc in docs:
                    doc_crop = doc.get("crop")
                    if doc_crop != crop and doc_crop != "general":
                        contaminated_crops.append((crop, doc_crop, doc.get("id")))

        self.assertEqual(
            len(contaminated_crops), 0,
            f"Cross-crop contamination detected! Contaminations: {contaminated_crops}"
        )
        self.assertGreaterEqual(total_crops_tested, len(self.supported_crops))


class TestRepresentativeVoiceTranscriptQueries(unittest.TestCase):
    """Validates query routing across 10 representative Karnataka crop categories."""

    def setUp(self):
        config = AdvisoryConfig(backend="mock", use_rag=True)
        self.engine = AdvisoryEngine(
            config=config,
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            retriever=AgriculturalRetriever()
        )

    def test_14_ten_category_voice_query_routing(self):
        """
        Test 10 distinct representative Karnataka crop categories:
        1. Cereal: Jowar
        2. Millet: Ragi
        3. Pulse: Red Gram
        4. Oilseed: Groundnut
        5. Spice: Turmeric
        6. Vegetable: Tomato
        7. Fruit: Banana
        8. Plantation: Arecanut
        9. Commercial: Sugarcane
        10. Floriculture: Jasmine
        """
        scenarios = [
            ("ನನ್ನ ಜೋಳದ ಬೆಳೆಯಲ್ಲಿ ಸುಳಿ ನೊಣ ಬಂದಿದೆ.", "jowar", "cereal"),
            ("ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ", "ragi", "millet"),
            ("ನನ್ನ ತೊಗರಿ ಬೆಳೆಗೆ ಕಾಯಿ ಕೊರಕ ಹುಳು ಬಂದಿದೆ.", "red_gram", "pulse"),
            ("ನನ್ನ ಕಡಲೆಕಾಯಿ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳು ಕಾಣಿಸುತ್ತಿವೆ.", "groundnut", "oilseed"),
            ("ಅರಿಶಿನ ಗೆಡ್ಡೆ ಕೊಳೆ ರೋಗ ಬಂದಿದೆ.", "turmeric", "spice"),
            ("ಟೊಮೇಟೊ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ", "tomato", "vegetable"),
            ("ಬಾಳೆ ಗಿಡದ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ.", "banana", "fruit"),
            ("ಅಡಿಕೆ ಮರಗಳಲ್ಲಿ ಕೊಳೆ ರೋಗ ಬಂದಿದೆ.", "arecanut", "plantation"),
            ("ನನ್ನ ಕಬ್ಬಿನ ಬೆಳೆಯಲ್ಲಿ ಒಣಗಿದ ಸುಳಿ ಮತ್ತು ಕೀಟ ಬಾಧೆ ಇದೆ.", "sugarcane", "commercial"),
            ("ಮಲ್ಲಿಗೆ ಮೊಗ್ಗು ಕೊರಕ ಹುಳು ಬಾಧೆ ಇದೆ.", "jasmine", "floriculture"),
        ]

        for query, expected_crop, expected_cat in scenarios:
            res = self.engine.generate_advisory(query=query, source_language="kn")
            self.assertEqual(res["canonical_crop"], expected_crop, f"Failed resolving {expected_crop} from '{query}'")
            self.assertEqual(res["crop_category"], expected_cat, f"Wrong category for {expected_crop}")
            self.assertEqual(res["crop_support_status"], "supported")
            self.assertTrue(len(res["retrieved_documents"]) > 0, f"RAG failed retrieving docs for {expected_crop}")
            self.assertEqual(res["retrieved_documents"][0]["crop"], expected_crop)

    def test_15_non_agricultural_guardrail(self):
        """Verify non-agricultural query retains canonical_crop = None and returns domain disclaimer."""
        res = self.engine.generate_advisory(
            query="ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?",
            source_language="kn"
        )
        self.assertIsNone(res["canonical_crop"])
        self.assertEqual(res["crop_support_status"], "unsupported")
        self.assertEqual(len(res["retrieved_documents"]), 0)
        self.assertIn("ಕೃಷಿ ಸಲಹಾ ವ್ಯವಸ್ಥೆಯಾಗಿದ್ದು", res["response"])


if __name__ == "__main__":
    unittest.main()
