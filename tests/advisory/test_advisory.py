"""
Unit Tests for RaithaMitra Agricultural Advisory Module.

Tests configuration validation, prompt formatting, language bridge contract,
mock backend responses, lazy-loading integrity, and advisory engine orchestration.
"""

import unittest
from typing import Dict, Any

from model.advisory.config import AdvisoryConfig, AdvisoryConfigError
from model.advisory.prompt_templates import (
    DEFAULT_AGRI_SYSTEM_PROMPT,
    format_messages,
    format_prompt
)
from model.advisory.language_bridge import (
    LanguageBridge,
    PassThroughLanguageBridge,
    LanguageBridgeError
)
from model.advisory.agriparam_engine import (
    AdvisoryError,
    AdvisoryValidationError,
    AdvisoryBackendError,
    AdvisoryBackend,
    MockAdvisoryBackend,
    AgriParamBackend,
    AdvisoryEngine
)
from model.advisory import generate_advisory, get_advisory_engine


class TestAdvisoryConfig(unittest.TestCase):
    """Test suite for AdvisoryConfig validation and defaults."""

    def test_default_config(self):
        config = AdvisoryConfig()
        config.validate()
        self.assertEqual(config.model_id, "bharatgenai/AgriParam")
        self.assertEqual(config.backend, "mock")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_new_tokens, 256)
        self.assertEqual(config.advisory_language, "en")
        self.assertTrue(config.trust_remote_code)

    def test_valid_custom_config(self):
        config = AdvisoryConfig(
            model_id="custom/agri-llm",
            backend="transformers",
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=512,
            advisory_language="hi"
        )
        config.validate()
        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.advisory_language, "hi")

    def test_invalid_backend_raises_error(self):
        config = AdvisoryConfig(backend="invalid_backend")
        with self.assertRaises(AdvisoryConfigError):
            config.validate()

    def test_invalid_temperature_raises_error(self):
        config = AdvisoryConfig(temperature=-0.5)
        with self.assertRaises(AdvisoryConfigError):
            config.validate()

        config2 = AdvisoryConfig(temperature=2.5)
        with self.assertRaises(AdvisoryConfigError):
            config2.validate()

    def test_invalid_tokens_raises_error(self):
        config = AdvisoryConfig(max_new_tokens=0)
        with self.assertRaises(AdvisoryConfigError):
            config.validate()

    def test_invalid_language_raises_error(self):
        config = AdvisoryConfig(advisory_language="french")
        with self.assertRaises(AdvisoryConfigError):
            config.validate()


class TestPromptTemplates(unittest.TestCase):
    """Test suite for prompt formatting utilities."""

    def test_default_system_prompt_present(self):
        self.assertTrue(len(DEFAULT_AGRI_SYSTEM_PROMPT) > 20)
        self.assertIn("RaithaMitra", DEFAULT_AGRI_SYSTEM_PROMPT)

    def test_format_messages_basic(self):
        messages = format_messages("What fertilizer is best for ragi?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_AGRI_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What fertilizer is best for ragi?")

    def test_format_messages_with_context(self):
        messages = format_messages(
            query="What fertilizer to use?",
            context="Crop: Ragi, Soil: Red sandy loam"
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("Context: Crop: Ragi, Soil: Red sandy loam", messages[1]["content"])
        self.assertIn("Question: What fertilizer to use?", messages[1]["content"])

    def test_format_prompt_string(self):
        prompt_str = format_prompt("How to control tomato leaf curl virus?")
        self.assertIn("<system_prompt>", prompt_str)
        self.assertIn("<user>", prompt_str)
        self.assertIn("How to control tomato leaf curl virus?", prompt_str)
        self.assertIn("<assistant>", prompt_str)


class TestLanguageBridge(unittest.TestCase):
    """Test suite for LanguageBridge interface and PassThrough implementation."""

    def setUp(self):
        self.bridge = PassThroughLanguageBridge()

    def test_passthrough_to_advisory_lang(self):
        text = "ನನ್ನ ಟೊಮೇಟೊ ಬೆಳೆಯ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ"
        res = self.bridge.translate_to_advisory_lang(text, source_lang="kn", target_lang="en")
        self.assertEqual(res, text)

    def test_passthrough_from_advisory_lang(self):
        text = "Use copper oxychloride spray"
        res = self.bridge.translate_from_advisory_lang(text, source_lang="en", target_lang="kn")
        self.assertEqual(res, text)

    def test_empty_translation_raises_error(self):
        with self.assertRaises(LanguageBridgeError):
            self.bridge.translate_to_advisory_lang("")

        with self.assertRaises(LanguageBridgeError):
            self.bridge.translate_from_advisory_lang("   ")


class TestMockAdvisoryBackend(unittest.TestCase):
    """Test suite for MockAdvisoryBackend."""

    def setUp(self):
        self.backend = MockAdvisoryBackend()

    def test_availability(self):
        self.assertTrue(self.backend.is_available())

    def test_tomato_query_response(self):
        resp = self.backend.generate("How to treat yellow leaves on my tomato crop?")
        self.assertIn("tomato", resp.lower())
        self.assertIn("copper oxychloride", resp.lower())

    def test_paddy_query_response(self):
        resp = self.backend.generate("Pest control for paddy stem borer")
        self.assertIn("paddy", resp.lower())

    def test_fallback_query_response(self):
        resp = self.backend.generate("Some unknown crop question")
        self.assertIn("Agricultural Advisory Recommendation", resp)


class TestAgriParamBackendLazyLoad(unittest.TestCase):
    """Test suite ensuring AgriParamBackend does NOT load model weights on instantiation."""

    def test_no_weight_loading_on_init(self):
        config = AdvisoryConfig(backend="transformers")
        backend = AgriParamBackend(config)
        # Weights must NOT be loaded upon class creation
        self.assertFalse(backend.is_loaded)
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._tokenizer)
        self.assertIsNone(backend._pipeline)
        self.assertTrue(backend.is_available())


class TestAdvisoryEngine(unittest.TestCase):
    """Test suite for AdvisoryEngine end-to-end orchestration."""

    def setUp(self):
        self.config = AdvisoryConfig(backend="mock")
        self.engine = AdvisoryEngine(config=self.config)

    def test_generate_advisory_schema(self):
        query = "ನನ್ನ ಟೊಮೇಟೊ ಬೆಳೆಯ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ"
        result = self.engine.generate_advisory(query, source_language="kn")

        self.assertIsInstance(result, dict)
        self.assertIn("query", result)
        self.assertIn("response", result)
        self.assertIn("intermediate_query", result)
        self.assertIn("intermediate_response", result)
        self.assertIn("source_language", result)
        self.assertIn("advisory_language", result)
        self.assertIn("target_language", result)
        self.assertIn("model", result)
        self.assertIn("backend", result)
        self.assertIn("processing_time_seconds", result)

        self.assertEqual(result["query"], query)
        self.assertEqual(result["source_language"], "kn")
        self.assertEqual(result["backend"], "mock")
        self.assertGreater(len(result["response"]), 10)
        self.assertGreaterEqual(result["processing_time_seconds"], 0.0)

    def test_empty_query_raises_validation_error(self):
        with self.assertRaises(AdvisoryValidationError):
            self.engine.generate_advisory("")

        with self.assertRaises(AdvisoryValidationError):
            self.engine.generate_advisory("   ")

        with self.assertRaises(AdvisoryValidationError):
            self.engine.generate_advisory(None)

    def test_custom_backend_injection(self):
        class CustomBackend(AdvisoryBackend):
            def generate(self, prompt: str, messages=None, **kwargs) -> str:
                return "Custom injected advice."
            def is_available(self) -> bool:
                return True

        custom_engine = AdvisoryEngine(backend=CustomBackend())
        result = custom_engine.generate_advisory("Test question")
        self.assertEqual(result["response"], "Custom injected advice.")

    def test_public_convenience_function(self):
        result = generate_advisory("How to cultivate ragi?")
        self.assertIn("response", result)
        self.assertIn("ragi", result["response"].lower())


if __name__ == "__main__":
    unittest.main()
