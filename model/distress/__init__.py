"""
RaithaMitra Distress Detection & Safe Escalation Module
=======================================================
Major Project: BAD685 — RaithaMitra
Department: Artificial Intelligence & Data Science, KSSEM Bengaluru
"""

from model.distress.config import (
    DistressLevel,
    DistressResult,
    DistressConfig,
    SAFETY_RESPONSE_KN,
    SAFETY_RESPONSE_EN,
    EMPATHY_PREFIX_KN,
    EMPATHY_PREFIX_EN,
    AGRICULTURAL_SUPPORT_REFERRAL_KN,
    AGRICULTURAL_SUPPORT_REFERRAL_EN,
)
from model.distress.detector import (
    DistressDetector,
    get_distress_detector,
)

__all__ = [
    "DistressLevel",
    "DistressResult",
    "DistressConfig",
    "DistressDetector",
    "get_distress_detector",
    "SAFETY_RESPONSE_KN",
    "SAFETY_RESPONSE_EN",
    "EMPATHY_PREFIX_KN",
    "EMPATHY_PREFIX_EN",
    "AGRICULTURAL_SUPPORT_REFERRAL_KN",
    "AGRICULTURAL_SUPPORT_REFERRAL_EN",
]
