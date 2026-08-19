"""
HTTP clients and provider interfaces for Karnataka APMC Mandi market price data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.parse
import json

logger = logging.getLogger(__name__)


class MarketClientError(Exception):
    """Raised when market data retrieval or communication fails."""
    pass


class BaseMarketClient(ABC):
    """Abstract base class for APMC / Mandi market data clients."""

    @abstractmethod
    def fetch_market_prices(
        self,
        commodity: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        state: str = "Karnataka"
    ) -> List[Dict[str, Any]]:
        """Fetch raw market price records from data provider."""
        raise NotImplementedError


class OfficialMarketClient(BaseMarketClient):
    """
    Client for official Government of India / Karnataka agricultural marketing endpoints
    (AGMARKNET / Data.gov.in / KSAMB).
    """

    DEFAULT_ENDPOINT = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 4.0
    ) -> None:
        self.endpoint_url = endpoint_url or self.DEFAULT_ENDPOINT
        self.api_key = api_key
        self.timeout = timeout

    def fetch_market_prices(
        self,
        commodity: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        state: str = "Karnataka"
    ) -> List[Dict[str, Any]]:
        """
        Queries official market price endpoint with timeout and safe failure handling.
        """
        params: Dict[str, str] = {
            "format": "json",
            "filters[state]": state,
        }
        if self.api_key:
            params["api-key"] = self.api_key
        if district:
            params["filters[district]"] = district
        if market:
            params["filters[market]"] = market
        if commodity:
            params["filters[commodity]"] = commodity

        url = f"{self.endpoint_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RaithaMitra-Advisory/1.0 (Agriculture Advisory Karnataka)",
                "Accept": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 200:
                    raise MarketClientError(f"Official market API returned HTTP {response.status}")
                payload = json.loads(response.read().decode("utf-8"))
                records = payload.get("records", [])
                return records
        except Exception as e:
            logger.warning("Official market API request failed: %s", e)
            raise MarketClientError(f"Official market data endpoint unavailable: {e}") from e


class MockMarketClient(BaseMarketClient):
    """
    Mock client used EXCLUSIVELY in automated tests for deterministic verification
    of Karnataka APMC market queries, date handling, and failure tolerances.
    """

    DEFAULT_TEST_RECORDS: List[Dict[str, Any]] = [
        {
            "commodity": "Ragi",
            "canonical_crop": "ragi",
            "market_name": "Mandya",
            "district": "Mandya",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 2800.0,
            "max_price": 3400.0,
            "modal_price": 3200.0,
            "unit": "₹/quintal",
            "arrivals": 45.0,
            "variety": "Local / Hybrid",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Maize",
            "canonical_crop": "maize",
            "market_name": "Belagavi",
            "district": "Belagavi",
            "state": "Karnataka",
            "market_date": "2026-08-18",  # Dated yesterday for testing date handling
            "min_price": 2100.0,
            "max_price": 2450.0,
            "modal_price": 2350.0,
            "unit": "₹/quintal",
            "arrivals": 120.0,
            "variety": "Hybrid Yellow",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Tomato",
            "canonical_crop": "tomato",
            "market_name": "Binny Mill (F&V)",
            "district": "Bengaluru Urban",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 1400.0,
            "max_price": 2200.0,
            "modal_price": 1800.0,
            "unit": "₹/quintal",
            "arrivals": 250.0,
            "variety": "Local / Hybrid",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Onion",
            "canonical_crop": "onion",
            "market_name": "Hubballi (Amaragol)",
            "district": "Dharwad",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 1800.0,
            "max_price": 2600.0,
            "modal_price": 2300.0,
            "unit": "₹/quintal",
            "arrivals": 320.0,
            "variety": "Red Big",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Onion",
            "canonical_crop": "onion",
            "market_name": "Yeshwanthpur",
            "district": "Bengaluru Urban",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 2000.0,
            "max_price": 2800.0,
            "modal_price": 2500.0,
            "unit": "₹/quintal",
            "arrivals": 410.0,
            "variety": "Bangalore Rose",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Paddy(Dhan)",
            "canonical_crop": "paddy",
            "market_name": "Mysuru (Bandipalya)",
            "district": "Mysuru",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 2250.0,
            "max_price": 2700.0,
            "modal_price": 2550.0,
            "unit": "₹/quintal",
            "arrivals": 85.0,
            "variety": "Sona Masuri",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        },
        {
            "commodity": "Groundnut",
            "canonical_crop": "groundnut",
            "market_name": "Chitradurga",
            "district": "Chitradurga",
            "state": "Karnataka",
            "market_date": "2026-08-19",
            "min_price": 5600.0,
            "max_price": 6800.0,
            "modal_price": 6400.0,
            "unit": "₹/quintal",
            "arrivals": 60.0,
            "variety": "Pod",
            "grade": "FAQ",
            "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            "source_url": "https://agmarknet.gov.in"
        }
    ]

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        should_fail: bool = False
    ) -> None:
        self.records = records if records is not None else list(self.DEFAULT_TEST_RECORDS)
        self.should_fail = should_fail

    def fetch_market_prices(
        self,
        commodity: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        state: str = "Karnataka"
    ) -> List[Dict[str, Any]]:
        """Filter mock records matching query parameters."""
        if self.should_fail:
            raise MarketClientError("Simulated market provider connection failure.")

        results = []
        for r in self.records:
            if state and r.get("state", "").lower() != state.lower():
                continue
            if district and district.lower() not in r.get("district", "").lower():
                continue
            if market and market.lower() not in r.get("market_name", "").lower():
                continue
            if commodity:
                crop_match = (
                    commodity.lower() in r.get("commodity", "").lower() or
                    commodity.lower() == r.get("canonical_crop", "").lower()
                )
                if not crop_match:
                    continue
            results.append(dict(r))

        return results
