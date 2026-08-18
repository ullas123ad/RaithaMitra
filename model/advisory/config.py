"""
Configuration dataclasses for the RaithaMitra Advisory Module.
Supports Dhenu2-1B, AgriParam, Mock backends, NLLB translation, and Local RAG.
"""

from dataclasses import dataclass, field
from typing import Optional, List

ALLOWED_BACKENDS = ["mock", "transformers", "dhenu", "remote"]
ALLOWED_LANGUAGES = ["kn", "en", "hi"]


class AdvisoryConfigError(Exception):
    """Raised when an invalid configuration is provided."""
    pass


@dataclass
class AdvisoryConfig:
    """
    Configuration settings for agricultural advisory inference and RAG.

    Attributes:
        model_id: Hugging Face model identifier (e.g. KissanAI/Dhenu2-In-Llama3.2-1B-Instruct).
        backend: Inference engine type ('mock', 'transformers', 'dhenu', 'remote').
        device: Device for local execution ('cpu', 'cuda').
        temperature: Sampling temperature for generation (0.0 to 1.0).
        top_p: Nucleus sampling parameter.
        repetition_penalty: Repetition penalty for generation.
        max_new_tokens: Maximum number of tokens generated in response.
        advisory_language: Internal language used by the LLM backend ('en' or 'hi').
        system_prompt: Custom system instructions override.
        cache_dir: Optional custom Hugging Face model cache directory.
        use_rag: Whether to retrieve relevant local agricultural knowledge.
        rag_top_k: Number of retrieved knowledge entries supplied to LLM context.
        rag_threshold: Minimum relevance score threshold for retrieved entries.
        rag_corpus_path: Optional custom path to agricultural_corpus.json.
    """
    model_id: str = "KissanAI/Dhenu2-In-Llama3.2-1B-Instruct"
    backend: str = "mock"
    device: str = "cpu"
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_new_tokens: int = 256
    advisory_language: str = "en"
    system_prompt: Optional[str] = None
    cache_dir: Optional[str] = None
    use_rag: bool = True
    rag_top_k: int = 3
    rag_threshold: float = 1.0
    rag_corpus_path: Optional[str] = None

    def validate(self) -> None:
        """Validates configuration parameters to avoid runtime execution failures."""
        if self.backend not in ALLOWED_BACKENDS:
            raise AdvisoryConfigError(
                f"Unsupported backend '{self.backend}'. Allowed: {ALLOWED_BACKENDS}"
            )

        if not (0.0 <= self.temperature <= 2.0):
            raise AdvisoryConfigError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if not (0.0 <= self.top_p <= 1.0):
            raise AdvisoryConfigError(
                f"top_p must be between 0.0 and 1.0, got {self.top_p}"
            )

        if self.max_new_tokens <= 0:
            raise AdvisoryConfigError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )

        if self.advisory_language not in ALLOWED_LANGUAGES:
            raise AdvisoryConfigError(
                f"Unsupported advisory language '{self.advisory_language}'. Allowed: {ALLOWED_LANGUAGES}"
            )

        if self.rag_top_k <= 0:
            raise AdvisoryConfigError(
                f"rag_top_k must be positive, got {self.rag_top_k}"
            )

        if self.rag_threshold < 0.0:
            raise AdvisoryConfigError(
                f"rag_threshold must be non-negative, got {self.rag_threshold}"
            )
