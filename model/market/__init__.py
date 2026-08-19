"""
Karnataka APMC Mandi Commodity Price Module for RaithaMitra.

Exports:
    MarketPriceRecord: Dataclass representing a single APMC market price record.
    MarketContext: Structured container for commodity market context.
    MarketService: Service for fetching and formatting APMC price records.
    BaseMarketClient: Abstract client interface.
    OfficialMarketClient: Live government endpoint client with failure tolerance.
    MockMarketClient: Mock client for deterministic unit testing.
    MarketClientError: Base exception for market client failures.
"""

from model.market.models import MarketPriceRecord, MarketContext
from model.market.client import (
    BaseMarketClient,
    OfficialMarketClient,
    MockMarketClient,
    MarketClientError,
)
from model.market.service import MarketService

__all__ = [
    "MarketPriceRecord",
    "MarketContext",
    "MarketService",
    "BaseMarketClient",
    "OfficialMarketClient",
    "MockMarketClient",
    "MarketClientError",
]
