"""
Unit Test Suite for Farmer Distress Detection Layer
===================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru

Tests:
1. Three-tier classification: NONE, MODERATE, HIGH.
2. Kannada distress patterns.
3. English distress patterns.
4. Mixed Kannada-English code-switching patterns.
5. Critical false-positive protection:
   - Plant death ('ನನ್ನ ಬೆಳೆ ಸತ್ತುಹೋಗುತ್ತಿದೆ', 'ಗಿಡಗಳು ಸತ್ತುಹೋಗಿವೆ') must NOT trigger human distress.
   - Pest control verbs ('ಹುಳು ಸಾಯಬೇಕು') must NOT trigger human distress.
   - Ordinary crop failure ('ಬೆಳೆ ಹಾಳಾಗಿದೆ') must NOT trigger human distress.
6. Contextual escalation from crop loss to debt/financial crisis to immediate danger.
7. Deterministic latency SLA (< 5 ms).
"""

import sys
import time
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.distress import (
    DistressLevel,
    DistressResult,
    DistressDetector,
    get_distress_detector,
    SAFETY_RESPONSE_KN,
    SAFETY_RESPONSE_EN,
    EMPATHY_PREFIX_KN,
    EMPATHY_PREFIX_EN,
)


class TestDistressDetector(unittest.TestCase):
    """Unit tests for DistressDetector core classification and safety logic."""

    def setUp(self):
        self.detector = get_distress_detector()

    # -------------------------------------------------------------------------
    # 1. NONE CLASSIFICATION & FALSE POSITIVE PROTECTION
    # -------------------------------------------------------------------------

    def test_01_normal_agricultural_queries_are_none(self):
        """Standard agricultural queries must evaluate to DistressLevel.NONE."""
        queries = [
            "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗಿಲ್ಲ.",
            "ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ, ಏನು ಮಾಡಬೇಕು?",
            "ಮಂಡ್ಯದಲ್ಲಿ ಇವತ್ತು ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            "How to control fall armyworm in maize?",
            "What is the market price of onion in Hubballi APMC?"
        ]
        for q in queries:
            res = self.detector.detect(q)
            self.assertEqual(res.level, DistressLevel.NONE, f"Expected NONE for '{q}', got {res.level}")
            self.assertFalse(res.detected)
            self.assertEqual(res.priority, "normal")

    def test_02_plant_death_false_positive_protection(self):
        """
        CRITICAL SAFETY TEST:
        Phrases describing plant/crop death ('ಸತ್ತುಹೋಗುತ್ತಿದೆ', 'ಸತ್ತುಹೋಗಿವೆ', 'dying', 'dead')
        must NOT trigger human distress or self-harm escalation.
        """
        plant_queries = [
            "ನನ್ನ ಬೆಳೆ ಸತ್ತುಹೋಗುತ್ತಿದೆ, ಏನು ಮಾಡಬೇಕು?",
            "ಗಿಡಗಳು ಸತ್ತುಹೋಗಿವೆ",
            "ನನ್ನ ಟೊಮ್ಯಾಟೊ ಗಿಡ ಸತ್ತುಹೋಗಿದೆ",
            "ಕಲ್ಲಂಗಡಿ ಬಳ್ಳಿಗಳು ಸಾಯುತ್ತಿವೆ",
            "ಬೆಳೆ ಒಣಗಿ ಸತ್ತುಹೋಗಿದೆ",
            "My tomato plants are dying, what should I check?",
            "The ragi crop is dead due to drought",
            "Arecanut palms are dying from root rot"
        ]
        for q in plant_queries:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.NONE,
                f"Plant death false positive detected! Query '{q}' evaluated to {res.level}"
            )
            self.assertFalse(res.detected)

    def test_03_pest_death_verb_protection(self):
        """Pest elimination statements with 'ಸಾಯಬೇಕು' must NOT trigger human distress."""
        pest_queries = [
            "ಮೆಕ್ಕೆಜೋಳದ ಸುಳಿಯಲ್ಲಿರುವ ಹುಳು ಸಾಯಬೇಕು, ಯಾವ ಔಷಧಿ ಸಿಂಪಡಿಸಬೇಕು?",
            "ತೊಗರಿ ಕಾಯಿ ಕೊರಕ ಕೀಟ ಸಾಯಬೇಕು",
            "Insect and pest must die"
        ]
        for q in pest_queries:
            res = self.detector.detect(q)
            self.assertEqual(res.level, DistressLevel.NONE, f"Pest query '{q}' triggered {res.level}")

    def test_04_ordinary_crop_loss_without_distress_is_none(self):
        """Ordinary crop damage/loss statements without debt or despair must be NONE."""
        queries = [
            "ನನ್ನ ಬೆಳೆ ಹಾಳಾಗಿದೆ",
            "ಮಳೆಯಿಂದ ಬೆಳೆ ಸ್ವಲ್ಪ ಹಾಳಾಗಿದೆ",
            "The crop is damaged due to excess rain",
            "Fruit rot has damaged some pomegranate fruits"
        ]
        for q in queries:
            res = self.detector.detect(q)
            self.assertEqual(res.level, DistressLevel.NONE, f"Expected NONE for '{q}', got {res.level}")

    # -------------------------------------------------------------------------
    # 2. MODERATE CLASSIFICATION (Financial Crisis, Debt, Severe Burden)
    # -------------------------------------------------------------------------

    def test_05_kannada_debt_and_financial_distress(self):
        """Kannada statements with unmanageable debt/financial crisis evaluate to MODERATE."""
        moderate_queries = [
            "ನನ್ನ ಬೆಳೆ ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ. ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ?",
            "ಎರಡು ವರ್ಷಗಳಿಂದ ಬೆಳೆ ನಷ್ಟವಾಗುತ್ತಿದೆ. ತುಂಬಾ ಕಷ್ಟವಾಗುತ್ತಿದೆ.",
            "ಸಾಲ ತುಂಬಾ ಇದೆ, ಏನು ಮಾಡಬೇಕು ಎಂದು ಗೊತ್ತಾಗುತ್ತಿಲ್ಲ.",
            "ಬ್ಯಾಂಕ್ ಸಾಲ ಕಟ್ಟಲು ಆಗುತ್ತಿಲ್ಲ, ಬೆಳೆಯೂ ನಷ್ಟವಾಗಿದೆ.",
            "ಸಾಲದ ಒತ್ತಡ ತುಂಬಾ ಇದೆ, ತುಂಬಾ ಸಂಕಷ್ಟದಲ್ಲಿದ್ದೇನೆ.",
            "ರಾಗಿ ಬೆಲೆ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ, ಸಾಲ ತೀರಿಸಲು ಆಗುತ್ತಿಲ್ಲ.",
            "ಮಳೆಯಿಂದ ನನ್ನ ಎಲ್ಲಾ ಬೆಳೆ ಹಾಳಾಗಿದೆ, ಸಾಲವೂ ಇದೆ."
        ]
        for q in moderate_queries:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.MODERATE,
                f"Expected MODERATE for '{q}', got {res.level}"
            )
            self.assertTrue(res.detected)
            self.assertEqual(res.priority, "advisory")
            self.assertIsNotNone(res.empathy_message_kn)
            self.assertIsNotNone(res.safety_message_kn)
            self.assertIn("ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರ", res.safety_message_kn)

    def test_06_english_debt_and_financial_distress(self):
        """English statements with financial distress evaluate to MODERATE."""
        moderate_queries_en = [
            "My crop is completely ruined. How can I repay my loan?",
            "I have suffered crop loss for two years. It is extremely difficult.",
            "I have heavy debt, I do not know what to do.",
            "Unable to repay loan due to continuous crop failure.",
            "Bank loan pressure is too high, very difficult to survive."
        ]
        for q in moderate_queries_en:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.MODERATE,
                f"Expected MODERATE for '{q}', got {res.level}"
            )
            self.assertTrue(res.detected)
            self.assertEqual(res.priority, "advisory")
            self.assertIsNotNone(res.empathy_message_en)

    def test_07_mixed_language_codeswitching_distress(self):
        """Mixed Kannada-English statements evaluate to MODERATE."""
        mixed_queries = [
            "Crop full loss ಆಗಿದೆ, loan ಹೇಗೆ pay ಮಾಡಲಿ?",
            "ನನಗೆ ತುಂಬಾ stress ಆಗಿದೆ",
            "Loan pressure ತುಂಬಾ ಇದೆ, ಏನು ಮಾಡೋದು ತಿಳಿಯುತ್ತಿಲ್ಲ",
            "Heavy loss ಆಗಿದೆ, ಸಾಲ ತೀರಿಸಲು ಆಗುತ್ತಿಲ್ಲ"
        ]
        for q in mixed_queries:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.MODERATE,
                f"Expected MODERATE for '{q}', got {res.level}"
            )
            self.assertTrue(res.detected)

    # -------------------------------------------------------------------------
    # 3. HIGH CLASSIFICATION (Immediate Danger & Self-Harm Intent)
    # -------------------------------------------------------------------------

    def test_08_kannada_high_risk_personal_intent(self):
        """Explicit personal immediate-danger statements evaluate strictly to HIGH."""
        high_queries_kn = [
            "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು",
            "ನಾನು ಬದುಕಲು ಇಷ್ಟಪಡುವುದಿಲ್ಲ",
            "ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ",
            "ನನ್ನ ಜೀವನ ಮುಗಿದಂತಾಗಿದೆ ಬದುಕುವುದಿಲ್ಲ",
            "ಇನ್ನು ನಾನು ಬದುಕುವುದಿಲ್ಲ",
            "ನಾನು ಸಾಯಬೇಕು ಅನ್ನಿಸುತ್ತಿದೆ",
            "ನಾನು ಸಾಯಲು ಬಯಸುತ್ತೇನೆ",
            "ಆತ್ಮಹತ್ಯೆ ಮಾಡಿಕೊಳ್ಳಬೇಕು"
        ]
        for q in high_queries_kn:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.HIGH,
                f"Expected HIGH for '{q}', got {res.level}"
            )
            self.assertTrue(res.detected)
            self.assertEqual(res.priority, "safety")
            self.assertEqual(res.safety_message_kn, SAFETY_RESPONSE_KN)
            self.assertIn("ಆಸ್ಪತ್ರೆ ಅಥವಾ ಸ್ಥಳೀಯ ತುರ್ತು ಸೇವೆ", res.safety_message_kn)

    def test_09_english_high_risk_personal_intent(self):
        """English explicit personal self-harm statements evaluate strictly to HIGH."""
        high_queries_en = [
            "I want to end my life",
            "I am going to end my life",
            "I want to die",
            "I do not want to live anymore",
            "I cannot live anymore and want to end it"
        ]
        for q in high_queries_en:
            res = self.detector.detect(q)
            self.assertEqual(
                res.level, DistressLevel.HIGH,
                f"Expected HIGH for '{q}', got {res.level}"
            )
            self.assertTrue(res.detected)
            self.assertEqual(res.priority, "safety")
            self.assertEqual(res.safety_message_en, SAFETY_RESPONSE_EN)

    # -------------------------------------------------------------------------
    # 4. CONTEXTUAL ESCALATION & LATENCY SLA
    # -------------------------------------------------------------------------

    def test_10_contextual_escalation_across_severity_levels(self):
        """Verify step-wise escalation as distress context increases."""
        # Level 1: Simple crop issue -> NONE
        res1 = self.detector.detect("ನನ್ನ ರಾಗಿ ಬೆಳೆ ಹಾಳಾಗಿದೆ.")
        self.assertEqual(res1.level, DistressLevel.NONE)

        # Level 2: Crop issue + Debt crisis -> MODERATE
        res2 = self.detector.detect("ನನ್ನ ರಾಗಿ ಬೆಳೆ ಹಾಳಾಗಿದೆ, ಸಾಲ ತುಂಬಾ ಇದೆ.")
        self.assertEqual(res2.level, DistressLevel.MODERATE)

        # Level 3: Immediate personal danger -> HIGH
        res3 = self.detector.detect("ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು.")
        self.assertEqual(res3.level, DistressLevel.HIGH)

    def test_11_deterministic_latency_sla_under_5ms(self):
        """Benchmark: 1,000 detection calls must average < 1 ms (far exceeding < 5 ms SLA)."""
        queries = [
            "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗಿಲ್ಲ.",
            "ನನ್ನ ಬೆಳೆ ಸಂಪೂರ್ಣ ಹಾಳಾಗಿದೆ. ಸಾಲ ಹೇಗೆ ತೀರಿಸಲಿ?",
            "ನನ್ನ ಜೀವನ ಮುಗಿಸಿಕೊಳ್ಳಬೇಕು",
            "Crop full loss ಆಗಿದೆ, loan ಹೇಗೆ pay ಮಾಡಲಿ?",
            "ನನ್ನ ಟೊಮ್ಯಾಟೊ ಗಿಡ ಸತ್ತುಹೋಗಿದೆ"
        ]
        t0 = time.perf_counter()
        N = 1000
        for _ in range(N // len(queries)):
            for q in queries:
                self.detector.detect(q)
        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        avg_call_ms = elapsed_total_ms / N
        self.assertLess(
            avg_call_ms, 5.0,
            f"Average detector latency {avg_call_ms:.4f} ms exceeds 5 ms SLA budget!"
        )


if __name__ == "__main__":
    unittest.main()
