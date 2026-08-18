"""
RaithaMitra Agricultural Advisory Module.

Provides LLM-driven agricultural recommendations for farmer queries,
with pluggable backends (Mock, AgriParam, Quantized, Remote) and a Language Bridge.
"""

from typing import Optional, Dict, Any

from model.advisory.config import (
    AdvisoryConfig,
    AdvisoryConfigError
)
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

__all__ = [
    "AdvisoryConfig",
    "AdvisoryConfigError",
    "DEFAULT_AGRI_SYSTEM_PROMPT",
    "format_messages",
    "format_prompt",
    "LanguageBridge",
    "PassThroughLanguageBridge",
    "LanguageBridgeError",
    "AdvisoryError",
    "AdvisoryValidationError",
    "AdvisoryBackendError",
    "AdvisoryBackend",
    "MockAdvisoryBackend",
    "AgriParamBackend",
    "AdvisoryEngine",
    "get_advisory_engine",
    "generate_advisory",
]

# Singleton instance
_ADVISORY_ENGINE: Optional[AdvisoryEngine] = None


def get_advisory_engine(
    config: Optional[AdvisoryConfig] = None,
    backend: Optional[AdvisoryBackend] = None,
    language_bridge: Optional[LanguageBridge] = None
) -> AdvisoryEngine:
    """Returns or initializes the singleton AdvisoryEngine instance."""
    global _ADVISORY_ENGINE
    if _ADVISORY_ENGINE is None or config is not None or backend is not None:
        _ADVISORY_ENGINE = AdvisoryEngine(
            config=config,
            backend=backend,
            language_bridge=language_bridge
        )
    return _ADVISORY_ENGINE


def generate_advisory(
    query: str,
    source_language: str = "kn",
    target_language: Optional[str] = None,
    config: Optional[AdvisoryConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Public API convenience function to generate agricultural advice for a query.

    Usage:
        from model.advisory import generate_advisory
        result = generate_advisory("ಟೊಮೇಟೊ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ")
        print(result["response"])
    """
    engine = get_advisory_engine(config=config)
    return engine.generate_advisory(
        query=query,
        source_language=source_language,
        target_language=target_language,
        **kwargs
    )
