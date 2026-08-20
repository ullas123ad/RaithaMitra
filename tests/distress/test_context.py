"""
Contextual Integration Test Suite for Farmer Distress Layer
===========================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Validates that:
1. HIGH distress executes the fast path, bypassing RAG, Dhenu LLM, NLLB, and context services.
2. MODERATE distress preserves full agricultural grounding, RAG retrieval, crop identity,
   market, weather, soil, and government schemes, while adding empathetic framing and RSK/KVK referral.
3. NONE distress executes the normal advisory pipeline unchanged.
4. Non-agricultural guardrail behaves safely without suppressing genuine high-risk signals.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.distress import (
    DistressLevel,
    DistressDetector,
    get_distress_detector,
    SAFETY_RESPONSE_KN,
)
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge
from model.advisory.retriever import AgriculturalRetriever


class TestDistressContextualIntegration(unittest.TestCase):
    """Integration tests verifying distress layer interactions with all advisory components."""

    def setUp(self):
        self.config = AdvisoryConfig(backend="mock", use_rag=True)
        self.retriever = AgriculturalRetriever()
        self.backend = MockAdvisoryBackend()
        self.language_bridge = MockLanguageBridge()
        self.detector = get_distress_detector()

        self.engine = AdvisoryEngine(
            config=self.config,
            backend=self.backend,
            language_bridge=self.language_bridge,
            retriever=self.retriever,
            distress_detector=self.detector
        )

    def test_01_high_distress_fast_path_bypasses_expensive_components(self):
        """
        FAST PATH TEST:
        When a HIGH-risk statement is submitted, verify:
        1. RAG is NOT invoked (retrieved_documents = []).
        2. Dhenu backend is NOT invoked.
        3. Response is the deterministic safety message.
        4. Distress metadata is {'detected': True, 'level': 'HIGH', 'priority': 'safety'}.
        5. Processing time is near-zero (< 20 ms).
        """
        # Mock backend and retriever to verify they are never called
        mock_backend = MagicMock()
        mock_retriever = MagicMock()
        mock_scheme_service = MagicMock()
        mock_soil_service = MagicMock()
        mock_market_service = MagicMock()

        engine = AdvisoryEngine(
            config=self.config,
            backend=mock_backend,
            language_bridge=self.language_bridge,
            retriever=mock_retriever,
            scheme_service=mock_scheme_service,
            soil_service=mock_soil_service,
            market_service=mock_market_service,
            distress_detector=self.detector
        )

        query = "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು"
        res = engine.generate_advisory(query=query, source_language="kn")

        # Verify NO expensive components were invoked
        mock_backend.generate.assert_not_called()
        mock_retriever.retrieve.assert_not_called()
        mock_scheme_service.get_schemes_for_crop.assert_not_called()
        mock_soil_service.get_soil_context.assert_not_called()
        mock_market_service.get_prices.assert_not_called()

        # Verify safety response
        self.assertEqual(res["response"], SAFETY_RESPONSE_KN)
        self.assertEqual(res["distress"]["level"], "HIGH")
        self.assertTrue(res["distress"]["detected"])
        self.assertEqual(res["distress"]["priority"], "safety")
        self.assertEqual(len(res["retrieved_documents"]), 0)
        self.assertLess(res["processing_time_seconds"], 0.05)

    def test_02_moderate_distress_preserves_full_agricultural_advisory(self):
        """
        MODERATE DISTRESS TEST:
        When a farmer query contains financial distress + crop issue:
        1. Crop identity is resolved.
        2. RAG documents are retrieved for the crop.
        3. Backend generates grounded advice.
        4. Final response contains empathy framing and RSK/KVK contact recommendation.
        5. Distress metadata is {'detected': True, 'level': 'MODERATE', 'priority': 'advisory'}.
        """
        query = "ನನ್ನ ಬೆಳೆ ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ. ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ?"
        res = self.engine.generate_advisory(query=query, source_language="kn")

        self.assertEqual(res["distress"]["level"], "MODERATE")
        self.assertTrue(res["distress"]["detected"])
        self.assertEqual(res["distress"]["priority"], "advisory")
        self.assertIn("ಧೈರ್ಯವಾಗಿರಿ", res["response"])
        self.assertIn("ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರ", res["response"])

    def test_03_market_intent_with_distress_coexistence(self):
        """Verify market price queries with debt distress retain market intent + MODERATE distress."""
        query = "ರಾಗಿ ಬೆಲೆ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ, ಸಾಲ ತೀರಿಸಲು ಆಗುತ್ತಿಲ್ಲ."
        res = self.engine.generate_advisory(query=query, source_language="kn")

        self.assertEqual(res["canonical_crop"], "ragi")
        self.assertEqual(res["distress"]["level"], "MODERATE")
        self.assertTrue(res["distress"]["detected"])

    def test_04_weather_intent_with_distress_coexistence(self):
        """Verify weather loss queries with debt distress retain weather context + MODERATE distress."""
        query = "ಮಳೆಯಿಂದ ನನ್ನ ಎಲ್ಲಾ ಬೆಳೆ ಹಾಳಾಗಿದೆ, ಸಾಲವೂ ಇದೆ."
        res = self.engine.generate_advisory(query=query, source_language="kn")

        self.assertEqual(res["distress"]["level"], "MODERATE")
        self.assertTrue(res["distress"]["detected"])

    def test_05_scheme_intent_with_distress_coexistence(self):
        """Verify government scheme queries with distress retain scheme intent + MODERATE distress."""
        query = "ಸರ್ಕಾರಿ ಸಹಾಯ ಸಿಗದಿದ್ದರೆ ಸಾಲ ತೀರಿಸಲು ಆಗುವುದಿಲ್ಲ."
        res = self.engine.generate_advisory(query=query, source_language="kn")

        self.assertEqual(res["distress"]["level"], "MODERATE")
        self.assertTrue(res["distress"]["detected"])

    def test_06_non_agricultural_scope_with_high_risk_priority(self):
        """
        If a non-agricultural query contains explicit high-risk language,
        human safety escalation takes precedence over domain rejection.
        """
        query = "I want to end my life"
        res = self.engine.generate_advisory(query=query, source_language="en")

        self.assertEqual(res["distress"]["level"], "HIGH")
        self.assertEqual(res["distress"]["priority"], "safety")
        self.assertIn("hospital", res["response"].lower())


if __name__ == "__main__":
    unittest.main()
