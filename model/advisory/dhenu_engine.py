"""
RaithaMitra Local Agricultural LLM Inference Engine.

Integrates KissanAI/Dhenu2-In-Llama3.2-1B-Instruct on CPU with lazy loading,
inference-mode context execution, and prompt structuring for ₹0 local compute.
"""

import time
from typing import Dict, Any, Optional, List

from model.advisory.config import AdvisoryConfig
from model.advisory.agriparam_engine import AdvisoryBackend, AdvisoryBackendError


class DhenuBackend(AdvisoryBackend):
    """
    Local CPU inference backend for KissanAI/Dhenu2-In-Llama3.2-1B-Instruct.
    Implements strict lazy loading to prevent memory allocations at import time.
    """

    def __init__(self, config: Optional[AdvisoryConfig] = None):
        """
        Initialize DhenuBackend with configuration.

        Args:
            config: AdvisoryConfig instance. If None, default AdvisoryConfig is used.
        """
        self.config = config or AdvisoryConfig(
            model_id="KissanAI/Dhenu2-In-Llama3.2-1B-Instruct",
            backend="dhenu"
        )
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded into memory."""
        return self._is_loaded

    def is_available(self) -> bool:
        """Returns True if the backend environment can run Dhenu2."""
        return True

    def load_model(self) -> None:
        """
        Lazily loads the Dhenu2-1B tokenizer and model weights on CPU in float32.
        Allocates memory only when explicitly called or on first inference request.
        """
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            device = getattr(self.config, "device", "cpu") or "cpu"
            cache_dir = getattr(self.config, "cache_dir", None)

            # 1. Load Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                clean_up_tokenization_spaces=False,
                cache_dir=cache_dir
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 2. Load Model in float32 for CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
            self._model.to(device)
            self._model.eval()

            # Optimize for 4 physical CPU cores
            torch.set_num_threads(4)

            self._is_loaded = True

        except Exception as e:
            raise AdvisoryBackendError(
                f"Failed to load Dhenu model '{self.config.model_id}': {str(e)}"
            )

    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generates agricultural recommendations using Dhenu2-1B on CPU.

        Args:
            prompt: Formatted plain-text prompt string.
            messages: Optional structured message list for chat templating.
            **kwargs: Generation parameter overrides.

        Returns:
            Generated agricultural response string.
        """
        if not self._is_loaded:
            self.load_model()

        try:
            import torch

            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", getattr(self.config, "max_new_tokens", 160)),
                "temperature": kwargs.get("temperature", getattr(self.config, "temperature", 0.7)),
                "top_p": kwargs.get("top_p", getattr(self.config, "top_p", 0.9)),
                "repetition_penalty": kwargs.get("repetition_penalty", getattr(self.config, "repetition_penalty", 1.15)),
                "do_sample": kwargs.get("temperature", getattr(self.config, "temperature", 0.7)) > 0.0,
                "use_cache": True,
                "num_beams": 1,
                "pad_token_id": self._tokenizer.eos_token_id
            }

            # Prepare prompt
            if messages and hasattr(self._tokenizer, "apply_chat_template"):
                input_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                input_text = prompt

            inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
            input_len = inputs.input_ids.shape[1]

            with torch.inference_mode():
                outputs = self._model.generate(**inputs, **gen_kwargs)
                response_tokens = outputs[0][input_len:]

            response_text = self._tokenizer.decode(
                response_tokens,
                skip_special_tokens=True
            ).strip()

            return response_text

        except Exception as e:
            raise AdvisoryBackendError(f"Dhenu inference execution error: {str(e)}")
