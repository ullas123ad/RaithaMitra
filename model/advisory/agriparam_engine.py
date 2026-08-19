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
        "pm-kisan": (
            "Under PM-KISAN, eligible landholding farmer families receive ₹6,000 per year in 3 equal installments of ₹2,000 via DBT. "
            "Farmers must complete mandatory eKYC and land seeding on pmkisan.gov.in or through their local Raitha Samparka Kendra."
        ),
        "installment": (
            "Under PM-KISAN, eligible landholding farmer families receive ₹6,000 per year in 3 equal installments of ₹2,000 via DBT. "
            "Farmers must complete mandatory eKYC and land seeding on pmkisan.gov.in or through their local Raitha Samparka Kendra."
        ),
        "insurance": (
            "Under Pradhan Mantri Fasal Bima Yojana (PMFBY) in Karnataka, farmers can insure notified crops through the Samrakshane portal. "
            "Premium is capped at 2% for Kharif food/oilseed crops and 1.5% for Rabi crops, with government subsidies covering the remainder."
        ),
        "drip": (
            "Under PMKSY Per Drop More Crop, assistance is available for micro-irrigation (drip/sprinkler). "
            "In Karnataka, combined subsidies reach up to 90% for SC/ST and 75% for general category farmers subject to official verification."
        ),
        "tractor": (
            "Under SMAM Farm Mechanization, subsidies are provided on approved tractors and implements for eligible farmers. "
            "Assistance is subject to state targets, empanelled models, and official verification at the local Raitha Samparka Kendra."
        ),
        "machinery": (
            "Under Sub-Mission on Agricultural Mechanization (SMAM) in Karnataka, subsidies of 40% to 50% for general farmers and "
            "50% to 90% for SC/ST farmers are available for approved farm equipment subject to department targets."
        ),
        "xyz": (
            "No verified government scheme was found matching this name. Please verify official schemes at your local Raitha Samparka Kendra "
            "or on the official Karnataka Agriculture portal (raitamitra.karnataka.gov.in)."
        ),
        "laptop": (
            "RaithaMitra is an agricultural advisory assistant dedicated to crop health, weather, soil, farming schemes, and market prices. Please ask an agriculture-related question."
        ),
        "repair": (
            "RaithaMitra is an agricultural advisory assistant dedicated to crop health, weather, soil, farming schemes, and market prices. Please ask an agriculture-related question."
        ),
        "saffron": (
            "Specific local package of practices for this crop is not currently available in the Karnataka agricultural knowledge base. Please consult your nearest Krishi Vigyan Kendra (KVK) or University of Agricultural Sciences for specialized guidance."
        ),
        "fertilizer": (
            "For balanced fertilizer management, do not apply arbitrary chemical dosages. Regional soils benefit from organic Farm Yard Manure (FYM). "
            "For crop-specific NPK dosages, obtain an official Soil Health Card test from your local Raitha Samparka Kendra."
        ),
        "soil": (
            "Regional soil health information provides general soil texture and classification. "
            "For field-specific nutrient diagnosis and Soil Health Card testing, visit your local Raitha Samparka Kendra (RSK)."
        ),
        "ragi price": (
            "At Mandya APMC on 2026-08-19, reported Ragi prices ranged from ₹2,800 to ₹3,400 per quintal with a modal price of ₹3,200/quintal (Arrivals: 45 tonnes). Note that actual price received depends on grain quality and moisture."
        ),
        "price of maize": (
            "At Belagavi APMC, latest available reported Maize price is dated 2026-08-18 with a modal price of ₹2,350/quintal (Range: ₹2,100 - ₹2,450/quintal, Arrivals: 120 tonnes)."
        ),
        "belagavi": (
            "At Belagavi APMC, latest available reported Maize price is dated 2026-08-18 with a modal price of ₹2,350/quintal (Range: ₹2,100 - ₹2,450/quintal, Arrivals: 120 tonnes)."
        ),
        "maize price": (
            "At Belagavi APMC, latest available reported Maize price is dated 2026-08-18 with a modal price of ₹2,350/quintal (Range: ₹2,100 - ₹2,450/quintal, Arrivals: 120 tonnes)."
        ),
        "tomato price": (
            "At Binny Mill (F&V) APMC in Bengaluru on 2026-08-19, reported Tomato prices ranged from ₹1,400 to ₹2,200 per quintal with a modal price of ₹1,800/quintal (Arrivals: 250 tonnes)."
        ),
        "onion price": (
            "Based on reported official APMC data for 2026-08-19, Yeshwanthpur Bengaluru reported a modal price of ₹2,500/quintal (Range: ₹2,000 - ₹2,800) and Hubballi APMC reported ₹2,300/quintal (Range: ₹1,800 - ₹2,600). Confirm current auction rates before transporting produce."
        ),
        "better price": (
            "Based on reported official APMC data for 2026-08-19, Yeshwanthpur Bengaluru reported a modal price of ₹2,500/quintal (Range: ₹2,000 - ₹2,800) and Hubballi APMC reported ₹2,300/quintal (Range: ₹1,800 - ₹2,600). Confirm current auction rates before transporting produce."
        ),
        "best price": (
            "Based on reported official APMC data for 2026-08-19, Yeshwanthpur Bengaluru reported a modal price of ₹2,500/quintal (Range: ₹2,000 - ₹2,800) and Hubballi APMC reported ₹2,300/quintal (Range: ₹1,800 - ₹2,600). Confirm current auction rates before transporting produce."
        ),
        "market price": (
            "Official APMC market prices vary by district and daily trading session. Please specify your crop and nearby APMC market to check latest reported prices."
        ),
        "scheme": (
            "Key agricultural schemes available in Karnataka include PM-KISAN (direct income support), PMFBY (crop insurance via Samrakshane), "
            "Krishi Bhagya (farm pond and water conservation subsidy), and KCC (concessional crop loans). Farmers can verify eligibility via the FRUITS portal."
        ),
        "drying": (
            "There has been very little rain and my ragi crop is drying. What should I do?"
        ),
        "drought": (
            "There has been very little rain and my ragi crop is drying. What should I do?"
        ),
        "holes": (
            "Maize fall armyworm requires prompt action: install pheromone traps @ 4/acre "
            "and apply Emamectin benzoate 5% SG (0.4g/L) inside the central whorl if holes appear."
        ),
        "armyworm": (
            "Maize fall armyworm requires prompt action: install pheromone traps @ 4/acre "
            "and apply Emamectin benzoate 5% SG (0.4g/L) inside the central whorl if holes appear."
        ),
        "curling": (
            "Chilli leaf curl virus is transmitted by whiteflies and thrips. "
            "Manage sucking pests using Neem oil (3ml/L) or Imidacloprid 17.8% SL (0.3ml/L)."
        ),
        "spots": (
            "Groundnut leaf spots (Tikka disease) can be managed with Mancozeb (2g/L) or Hexaconazole 5% EC (1ml/L)."
        ),
        "drainage": (
            "Excess rainfall and standing water cause root suffocation. Dig drainage trenches immediately "
            "and apply 1% urea foliar spray once stagnant water is cleared."
        ),
        "rainfall": (
            "Excess rainfall and standing water cause root suffocation. Dig drainage trenches immediately "
            "and apply 1% urea foliar spray once stagnant water is cleared."
        ),
        "rain": (
            "Excess rainfall and standing water cause root suffocation. Dig drainage trenches immediately "
            "and apply 1% urea foliar spray once stagnant water is cleared."
        ),
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
        )
    }

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        # Extract user query from messages if provided, or from prompt
        raw_text = prompt
        if messages:
            for m in messages:
                if m.get("role") == "user":
                    raw_text = m.get("content", "")
                    break

        if "Farmer Query:" in raw_text:
            query_only = raw_text.split("Farmer Query:")[-1].strip().lower()
        elif "Question:" in raw_text:
            query_only = raw_text.split("Question:")[-1].strip().lower()
        else:
            query_only = raw_text.strip().lower()

        # Single-word crop input without problem description -> ask for clarification
        short_crops = {"ragi", "paddy", "maize", "groundnut", "sugarcane", "cotton", "chilli", "onion", "potato", "banana", "tomato", "ರಾಗಿ", "ಭತ್ತ", "ಮೆಕ್ಕೆಜೋಳ", "ಕಡಲೆಕಾಯಿ", "ಕಬ್ಬು", "ಹತ್ತಿ", "ಮೆಣಸಿನಕಾಯಿ", "ಈರುಳ್ಳಿ", "ಆಲೂಗಡ್ಡೆ", "ಬಾಳೆ", "ಟೊಮ್ಯಾಟೊ"}
        clean_short = query_only.strip().rstrip(".?!")
        if clean_short in short_crops:
            return "What issue are you facing with your crop? Please specify if you need guidance regarding leaf symptoms, pests, diseases, irrigation, fertilizer/soil, market prices, or government schemes."

        for key, resp in self.MOCK_RESPONSES.items():
            if key in query_only:
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
        retriever: Optional[AgriculturalRetriever] = None,
        scheme_service: Optional[Any] = None,
        soil_service: Optional[Any] = None,
        market_service: Optional[Any] = None
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

        # Initialize Scheme Service
        if scheme_service is not None:
            self.scheme_service = scheme_service
        else:
            from model.schemes.service import SchemeService
            self.scheme_service = SchemeService()

        # Initialize Soil Service
        if soil_service is not None:
            self.soil_service = soil_service
        else:
            from model.soil.service import SoilService
            self.soil_service = SoilService()

        # Initialize Market Service
        if market_service is not None:
            self.market_service = market_service
        else:
            from model.market.service import MarketService
            self.market_service = MarketService()

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
        soil: Optional[Any] = None,
        market: Optional[Any] = None,
        crop: Optional[str] = None,
        schemes: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processes farmer query through Translation -> RAG Retrieval -> Scheme Retrieval -> Soil Context -> Market Context -> LLM Reasoning -> Translation.

        Args:
            query: The farmer's question in Kannada (or English/Hindi).
            source_language: Source language code (default 'kn').
            target_language: Output language code (default matches source_language).
            context: Optional explicit contextual metadata.
            location: Optional LocationContext instance.
            weather: Optional WeatherContext instance.
            soil: Optional SoilContext instance.
            market: Optional MarketContext instance.
            crop: Optional crop name string.
            schemes: Optional list of GovernmentScheme instances.

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

        # 3. Retrieve Relevant Government Schemes
        retrieved_schemes = []
        scheme_context_text = ""
        if schemes is not None:
            retrieved_schemes = schemes
            scheme_context_text = self.scheme_service.format_scheme_context(retrieved_schemes)
        elif self.scheme_service is not None:
            retrieved_schemes = self.scheme_service.find_relevant_schemes(
                query=clean_query,
                crop=canonical_crop,
                location=location
            )
            if not retrieved_schemes and intermediate_query != clean_query:
                # Also check translated query if raw query had no alias match
                retrieved_schemes = self.scheme_service.find_relevant_schemes(
                    query=intermediate_query,
                    crop=canonical_crop,
                    location=location
                )
            if retrieved_schemes:
                scheme_context_text = self.scheme_service.format_scheme_context(retrieved_schemes)

        # 4. Format Soil Health Context
        soil_context_text = ""
        soil_ctx = None
        if soil is not None:
            soil_ctx = soil
            if self.soil_service is not None:
                soil_context_text = self.soil_service.format_soil_context(soil_ctx)
        elif self.soil_service is not None and location is not None:
            soil_ctx = self.soil_service.get_soil_context(location=location, crop=canonical_crop)
            if soil_ctx and soil_ctx.available:
                soil_context_text = self.soil_service.format_soil_context(soil_ctx)

        # 5. Format APMC Mandi Market Context
        market_context_text = ""
        market_ctx = None
        is_market_query = any(k in clean_query.lower() or k in intermediate_query.lower() for k in ["ಬೆಲೆ", "price", "ಮಾರುಕಟ್ಟೆ", "mandi", "apmc", "ಮಾರಾಟ", "rate", "market", "cost"])
        if market is not None:
            market_ctx = market
            if self.market_service is not None:
                market_context_text = self.market_service.format_market_context(market_ctx)
        elif self.market_service is not None and (is_market_query or canonical_crop is not None):
            if is_market_query and canonical_crop:
                market_ctx = self.market_service.get_prices(crop=canonical_crop, location=location)
                if market_ctx and market_ctx.available:
                    market_context_text = self.market_service.format_market_context(market_ctx)

        # 6. Format Dynamic Weather / Location Context
        weather_context_text = ""
        if weather is not None:
            from model.weather.service import WeatherService
            weather_context_text = WeatherService().format_weather_context(weather)
        elif location is not None:
            weather_context_text = f"--- LOCALITY CONTEXT ---\nLocation: {getattr(location, 'hierarchy_label', str(location))}"

        # Combine explicit context, retrieved RAG context, government schemes, soil health, market, and dynamic weather context
        combined_context_parts = []
        if retrieved_context_text:
            combined_context_parts.append(retrieved_context_text)
        if scheme_context_text:
            combined_context_parts.append(scheme_context_text)
        if soil_context_text:
            combined_context_parts.append(soil_context_text)
        if market_context_text:
            combined_context_parts.append(market_context_text)
        if weather_context_text:
            combined_context_parts.append(weather_context_text)
        if context and context.strip():
            combined_context_parts.append(f"Farm Context: {context.strip()}")
        combined_context = "\n\n".join(combined_context_parts) if combined_context_parts else None

        # 7. Format Prompt and Generate Advisory via LLM Backend
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

        # 8. Translate Response back to Target Language (Kannada)
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
            "retrieved_schemes": [s.to_dict() if hasattr(s, "to_dict") else s for s in retrieved_schemes],
            "soil": soil_ctx.to_dict() if soil_ctx and hasattr(soil_ctx, "to_dict") else None,
            "market": market_ctx.to_dict() if market_ctx and hasattr(market_ctx, "to_dict") else None,
            "location": location.to_dict() if hasattr(location, "to_dict") else None,
            "weather": weather.to_dict() if hasattr(weather, "to_dict") else None,
            "retrieval_time_seconds": round(retrieval_latency, 4),
            "translation_in_time_seconds": round(t_tr_in, 4),
            "generation_time_seconds": round(t_gen, 4),
            "translation_out_time_seconds": round(t_tr_out, 4),
            "processing_time_seconds": round(total_time, 4)
        }
