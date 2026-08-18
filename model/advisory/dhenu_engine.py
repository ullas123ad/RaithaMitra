"""
RaithaMitra Dhenu Agricultural LLM Backend.

Integrates KissanAI/Dhenu2-In-Llama3.2-1B-Instruct as the primary local ₹0-cost
agricultural language model backend for CPU inference on resource-constrained devices.
"""

from typing import Optional, List, Dict, Any
import os

from model.advisory.config import AdvisoryConfig
from model.advisory.agriparam_engine import AdvisoryBackend, AdvisoryBackendError


class DhenuBackend(AdvisoryBackend):
    """
    Local CPU inference backend using KissanAI/Dhenu2-In-Llama3.2-1B-Instruct.
    Implements strict lazy loading to prevent model weight allocation upon import or instantiation.
    """

    def __init__(self, config: Optional[AdvisoryConfig] = None):
        self.config = config or AdvisoryConfig(
            backend="dhenu",
            model_id="KissanAI/Dhenu2-In-Llama3.2-1B-Instruct",
            device="cpu"
        )
        self._tokenizer = None
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model weights are currently loaded in RAM."""
        return self._is_loaded

    def is_available(self) -> bool:
        """Returns True if backend configuration is valid."""
        return bool(self.config.model_id)

    def load_model(self) -> None:
        """
        Lazily loads the Dhenu2-1B model and tokenizer into CPU memory.
        Loads once and reuses the model instance across inference requests.
        """
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            device = self.config.device or "cpu"

            # 1. Load Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                clean_up_tokenization_spaces=False,
                cache_dir=self.config.cache_dir
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 2. Load Model in float32 for CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=self.config.cache_dir
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
        Executes text generation using the loaded Dhenu model under torch.inference_mode.

        Args:
            prompt: Plaintext formatted prompt string.
            messages: Optional structured list of chat messages.
            **kwargs: Hyperparameter overrides (max_new_tokens, temperature, top_p).

        Returns:
            Generated agricultural advisory text.
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
                "do_sample": kwargs.get("temperature", self.config.temperature) > 0.0,
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
                generated_text = self._tokenizer.decode(
                    response_tokens,
                    skip_special_tokens=True
                )

            return generated_text.strip()

        except Exception as e:
            raise AdvisoryBackendError(f"Dhenu inference execution error: {str(e)}")
