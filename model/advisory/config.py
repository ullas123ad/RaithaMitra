"""
RaithaMitra Agricultural Advisory Configuration.

Defines configuration parameters for the LLM advisory backend,
including model identifiers, inference parameters, and execution modes.
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

    # Model identification
    model_id: str = "bharatgenai/AgriParam"

    # Backend selection: "mock" (default for development), "transformers", "quantized", "remote"
    backend: str = "mock"

    # Inference hyperparameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1

    # Execution device configuration
    device: Optional[str] = None  # None for auto-detect (cpu, cuda)
    trust_remote_code: bool = True
    cache_dir: Optional[str] = os.getenv("AGRIPARAM_CACHE_DIR", None)

    # Language configuration
    # Note: AgriParam natively supports English ("en") and Hindi ("hi").
    # Kannada queries are routed through the Language Bridge.
    advisory_language: str = "en"  # "en" or "hi"

    # Custom system prompt override (if None, standard prompt template is used)
    system_prompt: Optional[str] = None

    # Allowed backends
    ALLOWED_BACKENDS: List[str] = field(
        default_factory=lambda: ["mock", "transformers", "quantized", "remote"]
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
                f"advisory_language must be 'en' or 'hi' (AgriParam native languages), got '{self.advisory_language}'"
            )
