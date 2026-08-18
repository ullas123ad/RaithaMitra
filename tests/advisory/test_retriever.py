"""
Unit tests for RaithaMitra Agricultural Knowledge Retriever (RAG).
Validates cross-crop lexical retrieval, thresholding, determinism, schema integrity, and edge cases.
"""

import unittest
import os
import tempfile
import json
from typing import List, Dict, Any

from model.advisory.retriever import (
    AgriculturalRetriever,
    AgriculturalRetrieverError
)


class TestAgriculturalRetriever(unittest.TestCase):
    """Test suite for AgriculturalRetriever."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = AgriculturalRetriever()

    def test_corpus_loaded(self):
        """Verify that corpus is populated with entries."""
        self.assertGreaterEqual(self.retriever.corpus_size, 20)

    def test_1_paddy_query_retrieval(self):
        """TEST 1: Relevant paddy query returns paddy-related knowledge."""
        query = "Rice leaves are turning yellow. What should I check?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "paddy")
        self.assertIn("yellow", results[0]["title"].lower() + results[0]["content"].lower())

    def test_2_ragi_drought_query_retrieval(self):
        """TEST 2: Relevant ragi query returns ragi-related knowledge."""
        query = "There has been very little rain and my ragi crop is drying. What should I do?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "ragi")
        self.assertEqual(results[0]["topic"], "water_stress")
        self.assertIn("drought", results[0]["title"].lower() + results[0]["content"].lower())

    def test_3_maize_pest_query_retrieval(self):
        """TEST 3: Relevant maize pest query returns maize pest knowledge."""
        query = "Something is eating holes in my maize leaves. What should I check?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "maize")
        self.assertEqual(results[0]["topic"], "pest_management")
        self.assertIn("fall armyworm", results[0]["content"].lower())

    def test_4_groundnut_disease_query_retrieval(self):
        """TEST 4: Groundnut disease query returns relevant groundnut knowledge."""
        query = "My groundnut leaves have spots. What could be the reason?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "groundnut")
        self.assertEqual(results[0]["topic"], "disease_management")
        self.assertIn("tikka", results[0]["title"].lower())

    def test_5_cotton_pest_query_retrieval(self):
        """TEST 5: Cotton pest query returns cotton pest knowledge."""
        query = "I am seeing insects damaging my cotton crop. What should I check?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "cotton")
        self.assertEqual(results[0]["topic"], "pest_management")

    def test_6_chilli_curling_query_retrieval(self):
        """TEST 6: Chilli leaf problem returns chilli-related knowledge."""
        query = "My chilli leaves are curling. What should I check first?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "chilli")
        self.assertIn("leaf curl", results[0]["title"].lower())

    def test_7_irrelevant_query_does_not_return_unrelated_top_result(self):
        """TEST 7: Irrelevant query does not return an unrelated entry as highest-scoring result."""
        query = "How do I repair a broken smartphone battery and fix screen flickering?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertEqual(len(results), 0)

    def test_8_top_k_parameter(self):
        """TEST 8: top_k works correctly."""
        query = "paddy pest stem borer dead heart"
        results_1 = self.retriever.retrieve(query, top_k=1)
        results_3 = self.retriever.retrieve(query, top_k=3)
        self.assertEqual(len(results_1), 1)
        self.assertLessEqual(len(results_3), 3)

    def test_9_empty_query_handled_safely(self):
        """TEST 9: Empty query is handled safely."""
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])
        self.assertEqual(self.retriever.retrieve(None), [])

    def test_10_empty_corpus_handled_safely(self):
        """TEST 10: Empty corpus is handled safely."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump([], tf)
            temp_path = tf.name

        try:
            empty_retriever = AgriculturalRetriever(corpus_path=temp_path)
            res = empty_retriever.retrieve("ragi water stress")
            self.assertEqual(res, [])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_11_result_schema_completeness(self):
        """TEST 11: Results contain id, crop, topic, title, content, source, score."""
        query = "ragi drought moisture stress"
        results = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(len(results), 1)
        doc = results[0]

        expected_keys = {"id", "crop", "topic", "title", "content", "source", "score"}
        for k in expected_keys:
            self.assertIn(k, doc)
            self.assertIsNotNone(doc[k])

        self.assertIsInstance(doc["score"], float)
        self.assertGreater(doc["score"], 0.0)

    def test_12_determinism(self):
        """TEST 12: Same query produces deterministic results."""
        query = "There has been very little rain and my ragi crop is drying. What should I do?"
        run_1 = self.retriever.retrieve(query, top_k=3)
        run_2 = self.retriever.retrieve(query, top_k=3)
        self.assertEqual(run_1, run_2)

    def test_onion_excess_rain_retrieval(self):
        """Verify onion rainfall/disease query retrieval."""
        query = "My onion field received heavy rain. What problems should I watch for?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "onion")
        self.assertIn("rain", results[0]["title"].lower() + results[0]["content"].lower())

    def test_potato_lesions_retrieval(self):
        """Verify potato disease lesions query retrieval."""
        query = "There are dark lesions on my potato leaves. What should I check?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "potato")
        self.assertIn("blight", results[0]["title"].lower() + results[0]["content"].lower())

    def test_banana_yellowing_retrieval(self):
        """Verify banana nutrient yellowing query retrieval."""
        query = "My banana plants are showing yellow leaves. What should I check?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["crop"], "banana")
        self.assertIn("yellow", results[0]["title"].lower() + results[0]["content"].lower())

    def test_format_context(self):
        """Verify format_context returns clean structured text."""
        docs = [
            {
                "id": "test_001",
                "crop": "ragi",
                "topic": "water_stress",
                "title": "Ragi Drought Management",
                "content": "Apply life-saving irrigation.",
                "source": "UAS Bangalore",
                "score": 4.5
            }
        ]
        context_str = self.retriever.format_context(docs)
        self.assertIn("RETRIEVED AGRICULTURAL KNOWLEDGE", context_str)
        self.assertIn("Ragi", context_str)
        self.assertIn("UAS Bangalore", context_str)


if __name__ == "__main__":
    unittest.main()
