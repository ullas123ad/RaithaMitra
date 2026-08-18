"""
RaithaMitra Agricultural Advisory Engine and Backend Routing.

Provides modular orchestration across:
1. Language Translation Bridge (Kannada <-> English)
2. Local Agricultural Knowledge Retrieval (RAG from ICAR/UAS Corpus)
3. Advisory Backends (Dhenu2-1B, AgriParam, Mock)
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from model.advisory.config import AdvisoryConfig
from model.advisory.prompt_templates import (
    DEFAULT_AGRI_SYSTEM_PROMPT,
    format_messages,
    format_prompt
)
from model.advisory.language_bridge import (
    LanguageBridge,
    PassThroughLanguageBridge
)
from model.advisory.retriever import AgriculturalRetriever
from model.advisory.crop_identifier import resolve_canonical_crop


class AdvisoryError(Exception):
    """Base exception for advisory module errors."""
    pass


class AdvisoryValidationError(AdvisoryError):
    """Raised when query input or configuration is invalid."""
    pass


class AdvisoryBackendError(AdvisoryError):
    """Raised when the LLM inference backend encounters an execution failure."""
    pass


class AdvisoryBackend(ABC):
    """Abstract interface defining the contract for LLM inference backends."""

    @abstractmethod
    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """Generates advisory response text given a formatted prompt or messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend is ready for inference."""
        pass


class MockAdvisoryBackend(AdvisoryBackend):
    """
    Mock backend for fast, deterministic unit testing and CI validation.
    Returns targeted agricultural recommendations without loading large LLM weights.
    """

    MOCK_RESPONSES: Dict[str, str] = {
        "tomato": (
            "Tomato leaf yellowing can indicate early blight or nitrogen shortage. "
            "Inspect underside of leaves for dark spots and apply copper oxychloride (2.5g/L) "
            "if fungal infection is observed. Ensure soil is well-drained."
        ),
        "ragi": (
            "Ragi finger blast is managed by seed treatment with Carbendazim (2g/kg). "
            "Avoid excessive nitrogen fertilization and maintain clean weeding during tillering."
        ),
        "paddy": (
            "Paddy yellow stem borer can be managed using pheromone traps @ 5/acre. "
            "Apply Cartap hydrochloride 4G if dead hearts exceed 5% during vegetative stage."
        ),
        "maize": (
            "Maize fall armyworm requires prompt action: install pheromone traps @ 4/acre "
            "and apply Emamectin benzoate 5% SG (0.4g/L) inside the central whorl if holes appear."
        ),
        "drainage": (
            "Excess water causes root suffocation. Dig drainage trenches immediately "
            "and apply 1% urea foliar spray once stagnant water is cleared."
        ),
        "rain": (
            "Excess rainfall and standing water cause root suffocation. Dig drainage trenches immediately "
            "and apply 1% urea foliar spray once stagnant water is cleared."
        )
    }

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        p_lower = prompt.lower()
        for key, resp in self.MOCK_RESPONSES.items():
            if key in p_lower:
                return resp

        return (
            "Agricultural Advisory Recommendation: Maintain balanced NPK nutrition, "
            "monitor soil moisture, and consult your local Krishi Vigyan Kendra (KVK) "
            "for specific crop disease identification."
        )


class AgriParamBackend(AdvisoryBackend):
    """
    Optional Hugging Face transformers backend for AgriParam / external causal LLMs.
    Uses strict lazy loading so weights are never loaded on instantiation.
    """

    def __init__(self, config: AdvisoryConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def is_available(self) -> bool:
        return True

    def load_model(self) -> None:
        """Lazily load model weights into RAM/device."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self._model.to(self.config.device)
            self._model.eval()

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.config.device
            )
            self._is_loaded = True
        except Exception as e:
            raise AdvisoryBackendError(f"Failed to load AgriParam model '{self.config.model_id}': {str(e)}")

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        if not self._is_loaded:
            self.load_model()

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True if self.config.temperature > 0 else False
            )
            return outputs[0]["generated_text"]
        except Exception as e:
            raise AdvisoryBackendError(f"Generation error in AgriParam backend: {str(e)}")


class AdvisoryEngine:
    """
    Main orchestration engine for RaithaMitra Agricultural Advisory.
    Connects:
    1. Language Translation Bridge (Kannada -> English)
    2. Local Knowledge Retrieval (RAG)
    3. Agricultural LLM Backend (Dhenu2 / Mock / AgriParam)
    4. Language Translation Bridge (English -> Kannada)
    """

    def __init__(
        self,
        config: Optional[AdvisoryConfig] = None,
        backend: Optional[AdvisoryBackend] = None,
        language_bridge: Optional[LanguageBridge] = None,
        retriever: Optional[AgriculturalRetriever] = None
    ):
        self.config = config or AdvisoryConfig()
        self.config.validate()

        self.language_bridge = language_bridge or PassThroughLanguageBridge()

        # Initialize Retriever if RAG is enabled
        if retriever is not None:
            self.retriever = retriever
        elif self.config.use_rag:
            self.retriever = AgriculturalRetriever(
                corpus_path=self.config.rag_corpus_path,
                top_k=self.config.rag_top_k,
                relevance_threshold=self.config.rag_threshold
            )
        else:
            self.retriever = None

        if backend is not None:
            self.backend = backend
        elif self.config.backend == "mock":
            self.backend = MockAdvisoryBackend()
        elif self.config.backend == "dhenu":
            from model.advisory.dhenu_engine import DhenuBackend
            self.backend = DhenuBackend(config=self.config)
        elif self.config.backend == "transformers":
            self.backend = AgriParamBackend(config=self.config)
        else:
            raise AdvisoryConfigError(f"Backend '{self.config.backend}' is not implemented.")

    def generate_advisory(
        self,
        query: str,
        source_language: str = "kn",
        target_language: Optional[str] = None,
        context: Optional[str] = None,
        location: Optional[Any] = None,
        weather: Optional[Any] = None,
        crop: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processes farmer query through Translation -> RAG Retrieval -> LLM Reasoning -> Translation.

        Args:
            query: The farmer's question in Kannada (or English/Hindi).
            source_language: Source language code (default 'kn').
            target_language: Output language code (default matches source_language).
            context: Optional explicit contextual metadata.
            location: Optional LocationContext instance.
            weather: Optional WeatherContext instance.
            crop: Optional crop name string.

        Returns:
            Dictionary containing structured advisory output and performance metrics.
        """
        if not query or not query.strip():
            raise AdvisoryValidationError("Farmer query text cannot be empty.")

        target_lang = target_language or source_language
        clean_query = query.strip()
        t_start = time.time()

        # 1. Translate Query from Source Language to Internal Advisory Language (English)
        t_tr_in_0 = time.time()
        intermediate_query = self.language_bridge.translate_to_advisory_lang(
            clean_query,
            source_lang=source_language,
            target_lang=self.config.advisory_language
        )
        t_tr_in = time.time() - t_tr_in_0

        # Resolve Canonical Crop Identity (priority: explicit arg -> original Kannada query -> translated query)
        canonical_crop = resolve_canonical_crop(
            query=clean_query,
            translated_query=intermediate_query,
            explicit_crop=crop
        )

        # 2. Retrieve Relevant Local Agricultural Knowledge (RAG)
        retrieved_docs = []
        retrieval_latency = 0.0
        retrieved_context_text = ""

        if self.config.use_rag and self.retriever is not None:
            t_rag_0 = time.time()
            retrieved_docs = self.retriever.retrieve(
                intermediate_query,
                crop=canonical_crop,
                top_k=self.config.rag_top_k
            )
            retrieval_latency = time.time() - t_rag_0
            retrieved_context_text = self.retriever.format_context(retrieved_docs)

        # 3. Format Dynamic Weather / Location Context
        weather_context_text = ""
        if weather is not None:
            from model.weather.service import WeatherService
            weather_context_text = WeatherService().format_weather_context(weather)
        elif location is not None:
            weather_context_text = f"--- LOCALITY CONTEXT ---\nLocation: {getattr(location, 'hierarchy_label', str(location))}"

        # Combine explicit context, retrieved RAG context, and dynamic weather context
        combined_context_parts = []
        if retrieved_context_text:
            combined_context_parts.append(retrieved_context_text)
        if weather_context_text:
            combined_context_parts.append(weather_context_text)
        if context and context.strip():
            combined_context_parts.append(f"Farm Context: {context.strip()}")
        combined_context = "\n\n".join(combined_context_parts) if combined_context_parts else None

        # 4. Format Prompt and Generate Advisory via LLM Backend
        messages = format_messages(
            query=intermediate_query,
            context=combined_context,
            system_prompt=self.config.system_prompt
        )
        prompt_str = format_prompt(
            query=intermediate_query,
            context=combined_context,
            system_prompt=self.config.system_prompt
        )

        t_gen_0 = time.time()
        intermediate_response = self.backend.generate(
            prompt=prompt_str,
            messages=messages,
            **kwargs
        )
        t_gen = time.time() - t_gen_0

        # 5. Translate Response back to Target Language (Kannada)
        t_tr_out_0 = time.time()
        final_response = self.language_bridge.translate_from_advisory_lang(
            intermediate_response,
            source_lang=self.config.advisory_language,
            target_lang=target_lang
        )
        t_tr_out = time.time() - t_tr_out_0

        total_time = time.time() - t_start

        return {
            "query": clean_query,
            "response": final_response,
            "canonical_crop": canonical_crop,
            "intermediate_query": intermediate_query,
            "intermediate_response": intermediate_response,
            "source_language": source_language,
            "advisory_language": self.config.advisory_language,
            "target_language": target_lang,
            "model": self.config.model_id,
            "backend": self.config.backend,
            "rag_enabled": self.config.use_rag,
            "retrieved_documents": retrieved_docs,
            "location": location.to_dict() if hasattr(location, "to_dict") else None,
            "weather": weather.to_dict() if hasattr(weather, "to_dict") else None,
            "retrieval_time_seconds": round(retrieval_latency, 4),
            "translation_in_time_seconds": round(t_tr_in, 4),
            "generation_time_seconds": round(t_gen, 4),
            "translation_out_time_seconds": round(t_tr_out, 4),
            "processing_time_seconds": round(total_time, 4)
        }
