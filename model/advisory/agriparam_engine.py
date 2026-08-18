"""
RaithaMitra Agricultural Advisory Engine and Backend Abstraction.

Provides a decoupled, extensible advisory engine supporting multiple LLM backends:
1. MockAdvisoryBackend: Fast, deterministic testing without downloading model weights.
2. DhenuBackend: Local 1B CPU Agricultural LLM (KissanAI/Dhenu2-In-Llama3.2-1B-Instruct).
3. AgriParamBackend: Official bharatgenai/AgriParam Hugging Face integration with lazy loading.
4. Replaceable interface for future Quantized and Remote backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import os

from model.advisory.config import AdvisoryConfig, AdvisoryConfigError
from model.advisory.prompt_templates import format_messages, format_prompt, DEFAULT_AGRI_SYSTEM_PROMPT
from model.advisory.language_bridge import LanguageBridge, PassThroughLanguageBridge, LanguageBridgeError


class AdvisoryError(Exception):
    """Base exception for all advisory module errors."""
    pass


class AdvisoryValidationError(AdvisoryError):
    """Raised when input validation fails (e.g. empty or invalid query)."""
    pass


class AdvisoryBackendError(AdvisoryError):
    """Raised when backend model execution encounters an error."""
    pass


# =====================================================================
# Backend Abstraction Interface
# =====================================================================

class AdvisoryBackend(ABC):
    """Abstract interface for all LLM advisory backends."""

    @abstractmethod
    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """
        Generates an agricultural advisory response from the model.

        Args:
            prompt: Plaintext formatted prompt string.
            messages: Optional structured list of chat messages.
            **kwargs: Generation parameter overrides.

        Returns:
            Generated response string.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend is ready and available for generation."""
        pass


# =====================================================================
# Mock Advisory Backend (For Fast, Deterministic Development & Testing)
# =====================================================================

class MockAdvisoryBackend(AdvisoryBackend):
    """
    Lightweight mock backend returning structured agricultural recommendations
    without downloading or loading large model weights.
    """

    # Domain response patterns for common agricultural test topics
    MOCK_KNOWLEDGE: Dict[str, str] = {
        "tomato": (
            "For yellowing leaves in tomato plants, check for Early Blight or Nitrogen deficiency. "
            "Recommendation: Ensure proper soil drainage, avoid overhead watering, "
            "and apply copper oxychloride spray (2g/L) or balanced NPK (19:19:19) fertilizer."
        ),
        "paddy": (
            "For pest infestation in paddy crops (e.g., Stem Borer / Leaf Folder), "
            "Recommendation: Maintain optimum water levels, use pheromone traps (5/acre), "
            "and apply Chlorantraniliprole 18.5% SC @ 60 ml/acre if pest population exceeds threshold."
        ),
        "ragi": (
            "Ragi (Finger Millet) advisory: For blast disease resistance, ensure seed treatment "
            "with Carbendazim (2g/kg). Maintain 25x10 cm spacing and apply recommended FYM and fertilizers."
        ),
        "onion": (
            "Onion crop advisory: To manage Purple Blotch and Thrips, apply Mancozeb 75% WP (2.5g/L) "
            "mixed with a sticker, and ensure soil is not waterlogged."
        ),
        "weather": (
            "Weather Advisory: Monitor rainfall forecasts closely before applying chemical sprays or irrigation. "
            "Ensure adequate field drainage during heavy monsoon spells."
        ),
        "rainfall": (
            "Excessive rainfall can lead to waterlogging, root rot, and increased fungal infection risk. "
            "Ensure immediate field drainage and avoid fertilizer application during heavy downpours."
        ),
        "scheme": (
            "Government Scheme Advisory: Farmers can apply for state and central subsidies via the "
            "official Krishi / Seva Sindhu portals or visit the nearest Krishi Vigyan Kendra (KVK)."
        )
    }

    DEFAULT_RESPONSE: str = (
        "Agricultural Advisory Recommendation: Ensure proper soil moisture, inspect crops for early "
        "pest symptoms, and follow integrated nutrient management practices suitable for your local agro-climatic zone."
    )

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """Returns a deterministic, domain-relevant mock agricultural advisory."""
        query_text = prompt.lower()
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    query_text += " " + msg.get("content", "").lower()

        # Match domain keywords
        for keyword, advice in self.MOCK_KNOWLEDGE.items():
            if keyword in query_text:
                return advice

        return self.DEFAULT_RESPONSE

    def is_available(self) -> bool:
        return True


# =====================================================================
# AgriParam LLM Backend (Official Transformers Integration)
# =====================================================================

class AgriParamBackend(AdvisoryBackend):
    """
    Official Hugging Face Transformers wrapper for bharatgenai/AgriParam.
    Uses lazy loading so model weights are NEVER loaded upon import or instantiation.
    """

    def __init__(self, config: Optional[AdvisoryConfig] = None):
        self.config = config or AdvisoryConfig(
            backend="transformers",
            model_id="bharatgenai/AgriParam"
        )
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns whether model weights are currently loaded in memory."""
        return self._is_loaded

    def is_available(self) -> bool:
        """Returns True if the backend is configured."""
        return bool(self.config.model_id)

    def load_model(self) -> None:
        """
        Lazily loads the AgriParam model and tokenizer into memory.
        Explicitly requires trust_remote_code=True per official specifications.
        """
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            device = self.config.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch_dtype,
                cache_dir=self.config.cache_dir
            )
            self._model.to(device)
            self._model.eval()

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if device == "cuda" else -1
            )
            self._is_loaded = True

        except Exception as e:
            raise AdvisoryBackendError(
                f"Failed to load AgriParam model '{self.config.model_id}': {str(e)}"
            )

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """
        Executes text generation using the loaded AgriParam model.
        """
        if not self._is_loaded:
            self.load_model()

        try:
            import torch

            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
                "do_sample": self.config.temperature > 0.0,
                "pad_token_id": self._tokenizer.eos_token_id if self._tokenizer else None
            }

            with torch.inference_mode():
                if messages and hasattr(self._tokenizer, "apply_chat_template"):
                    input_text = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    input_text = prompt

                inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
                outputs = self._model.generate(**inputs, **gen_kwargs)
                input_len = inputs.input_ids.shape[1]
                response_tokens = outputs[0][input_len:]
                generated_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            raise AdvisoryBackendError(f"AgriParam inference error: {str(e)}")


# =====================================================================
# Advisory Engine Coordinator
# =====================================================================

class AdvisoryEngine:
    """
    Coordinates the end-to-end agricultural advisory flow:
    1. Input query validation
    2. Cross-lingual translation via LanguageBridge
    3. Context and prompt formatting
    4. Execution against pluggable AdvisoryBackend (Dhenu, AgriParam, Mock)
    5. Reverse translation to Kannada (when configured)
    6. Structured payload packaging
    """

    def __init__(
        self,
        config: Optional[AdvisoryConfig] = None,
        backend: Optional[AdvisoryBackend] = None,
        language_bridge: Optional[LanguageBridge] = None
    ):
        self.config = config or AdvisoryConfig()
        self.config.validate()

        # Pluggable Language Bridge
        self.language_bridge = language_bridge or PassThroughLanguageBridge()

        # Pluggable Backend
        if backend is not None:
            self.backend = backend
        elif self.config.backend == "mock":
            self.backend = MockAdvisoryBackend()
        elif self.config.backend == "dhenu":
            from model.advisory.dhenu_engine import DhenuBackend
            self.backend = DhenuBackend(self.config)
        elif self.config.backend == "transformers":
            self.backend = AgriParamBackend(self.config)
        else:
            raise AdvisoryConfigError(
                f"Backend '{self.config.backend}' is not yet implemented. Use 'mock', 'dhenu', or 'transformers'."
            )

    def generate_advisory(
        self,
        query: str,
        source_language: str = "kn",
        target_language: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an agricultural advisory response for a farmer query.

        Args:
            query: Farmer question text.
            source_language: Language code of input query (default: 'kn').
            target_language: Language code of desired final response (default: same as source).
            context: Optional domain context (crop, season, location).
            **kwargs: Generation parameter overrides.

        Returns:
            Structured advisory dictionary payload.
        """
        # 1. Input validation
        if query is None or not isinstance(query, str) or not query.strip():
            raise AdvisoryValidationError("Farmer query must be a non-empty string.")

        clean_query = query.strip()
        final_lang = target_language or source_language
        advisory_lang = self.config.advisory_language

        start_time = time.time()

        # 2. Bridge query: Kannada -> Advisory language (English)
        intermediate_query = self.language_bridge.translate_to_advisory_lang(
            clean_query,
            source_lang=source_language,
            target_lang=advisory_lang
        )

        # 3. Format prompt
        messages = format_messages(
            query=intermediate_query,
            system_prompt=self.config.system_prompt,
            context=context
        )
        prompt_str = format_prompt(
            query=intermediate_query,
            system_prompt=self.config.system_prompt,
            context=context
        )

        # 4. Generate response via pluggable backend (Dhenu, etc.)
        llm_response = self.backend.generate(
            prompt=prompt_str,
            messages=messages,
            **kwargs
        )

        # 5. Bridge response: Advisory language -> Final target language
        final_response = self.language_bridge.translate_from_advisory_lang(
            llm_response,
            source_lang=advisory_lang,
            target_lang=final_lang
        )

        elapsed_time = time.time() - start_time

        return {
            "query": clean_query,
            "response": final_response,
            "intermediate_query": intermediate_query,
            "intermediate_response": llm_response,
            "source_language": source_language,
            "advisory_language": advisory_lang,
            "target_language": final_lang,
            "model": self.config.model_id,
            "backend": self.config.backend,
            "processing_time_seconds": round(elapsed_time, 4)
        }
