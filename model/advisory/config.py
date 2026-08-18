"""
RaithaMitra Agricultural Advisory Configuration.

Defines configuration parameters for the LLM advisory backends:
1. Dhenu (KissanAI/Dhenu2-In-Llama3.2-1B-Instruct) - Local 1B CPU Agricultural LLM
2. AgriParam (bharatgenai/AgriParam) - Optional remote/high-memory LLM
3. Mock Backend - Fast, deterministic testing
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


class AdvisoryConfigError(Exception):
    """Raised when advisory configuration parameters are invalid."""
    pass


@dataclass
class AdvisoryConfig:
    """Configuration settings for RaithaMitra Agricultural Advisory Engine."""

    # Backend selection: "mock" (default for testing), "dhenu", "transformers", "quantized", "remote"
    backend: str = "mock"

    # Model identification
    model_id: str = "KissanAI/Dhenu2-In-Llama3.2-1B-Instruct"

    # Inference hyperparameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1

    # Execution device configuration
    device: Optional[str] = "cpu"
    trust_remote_code: bool = False
    cache_dir: Optional[str] = os.getenv("HF_HOME", None)

    # Language configuration
    # Note: Dhenu and AgriParam operate natively in English ("en").
    # Kannada queries are routed through the Language Bridge.
    advisory_language: str = "en"  # "en" or "hi"

    # Custom system prompt override (if None, standard prompt template is used)
    system_prompt: Optional[str] = None

    # Allowed backends
    ALLOWED_BACKENDS: List[str] = field(
        default_factory=lambda: ["mock", "dhenu", "transformers", "quantized", "remote"]
    )

    def validate(self) -> None:
        """Validates configuration parameters."""
        if not self.model_id or not isinstance(self.model_id, str):
            raise AdvisoryConfigError("model_id must be a non-empty string.")

        if self.backend not in self.ALLOWED_BACKENDS:
            raise AdvisoryConfigError(
                f"Invalid backend '{self.backend}'. Allowed backends: {self.ALLOWED_BACKENDS}"
            )

        if not (0.0 <= self.temperature <= 2.0):
            raise AdvisoryConfigError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if not (0.0 < self.top_p <= 1.0):
            raise AdvisoryConfigError(
                f"top_p must be between 0.0 (exclusive) and 1.0, got {self.top_p}"
            )

        if self.max_new_tokens <= 0:
            raise AdvisoryConfigError(
                f"max_new_tokens must be greater than 0, got {self.max_new_tokens}"
            )

        if self.advisory_language not in ["en", "hi"]:
            raise AdvisoryConfigError(
                f"advisory_language must be 'en' or 'hi' (Advisory LLM native languages), got '{self.advisory_language}'"
            )
