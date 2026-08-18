"""
RaithaMitra Agricultural Advisory Module.

Provides LLM-driven agricultural recommendations for farmer queries,
with pluggable backends (Dhenu2-1B, AgriParam, Mock) and Language Bridges (NLLB-200, Mock, PassThrough).
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
    MockLanguageBridge,
    NLLBTranslationBridge,
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
from model.advisory.dhenu_engine import DhenuBackend

__all__ = [
    "AdvisoryConfig",
    "AdvisoryConfigError",
    "DEFAULT_AGRI_SYSTEM_PROMPT",
    "format_messages",
    "format_prompt",
    "LanguageBridge",
    "PassThroughLanguageBridge",
    "MockLanguageBridge",
    "NLLBTranslationBridge",
    "LanguageBridgeError",
    "AdvisoryError",
    "AdvisoryValidationError",
    "AdvisoryBackendError",
    "AdvisoryBackend",
    "MockAdvisoryBackend",
    "AgriParamBackend",
    "DhenuBackend",
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
    if _ADVISORY_ENGINE is None or config is not None or backend is not None or language_bridge is not None:
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
    language_bridge: Optional[LanguageBridge] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Public API convenience function to generate agricultural advice for a query.

    Usage:
        from model.advisory import generate_advisory
        result = generate_advisory("ನನ್ನ ಟೊಮೇಟೊ ಗಿಡದ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ. ಏನು ಮಾಡಬೇಕು?")
        print(result["response"])
    """
    engine = get_advisory_engine(config=config, language_bridge=language_bridge)
    return engine.generate_advisory(
        query=query,
        source_language=source_language,
        target_language=target_language,
        **kwargs
    )
