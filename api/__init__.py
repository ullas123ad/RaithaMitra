"""
RaithaMitra HTTP Backend API Package
====================================
Exposes the multi-module agricultural advisory orchestration as a clean REST API.
"""

from api.app import create_app

__all__ = ["create_app"]
