"""
Karnataka Soil Health Context Module for RaithaMitra.

Exports:
    SoilContext: Structured dataclass representing soil profile and laboratory test data.
    SoilService: Service for retrieving and formatting Karnataka soil context.
    SoilServiceError: Base exception for soil service operations.
"""

from model.soil.models import SoilContext
from model.soil.service import SoilService, SoilServiceError

__all__ = [
    "SoilContext",
    "SoilService",
    "SoilServiceError",
]
