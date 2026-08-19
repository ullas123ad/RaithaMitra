"""
Karnataka APMC Mandi Market Price Service for RaithaMitra.

Provides structured, authoritative commodity price context from AGMARKNET and
Karnataka APMCs with strict anti-speculation and truth-in-dating rules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from model.location.models import LocationContext
from model.advisory.crop_identifier import normalize_crop_name
from model.market.client import BaseMarketClient, OfficialMarketClient, MarketClientError
from model.market.models import MarketPriceRecord, MarketContext

logger = logging.getLogger(__name__)


# Mapping canonical crops to official AGMARKNET commodity names and query aliases
CROP_COMMODITY_MAP: Dict[str, List[str]] = {
    "ragi": ["Ragi", "Finger Millet", "ರಾಗಿ"],
    "paddy": ["Paddy(Dhan)", "Paddy", "Rice", "ಭತ್ತ"],
    "maize": ["Maize", "Corn", "ಮೆಕ್ಕೆಜೋಳ"],
    "groundnut": ["Groundnut", "Peanut", "ಕಡಲೆಕಾಯಿ"],
    "sugarcane": ["Sugarcane", "ಕಬ್ಬು"],
    "cotton": ["Cotton", "ಹತ್ತಿ"],
    "chilli": ["Chilli Red", "Green Chilli", "ಮೆಣಸಿನಕಾಯಿ"],
    "onion": ["Onion", "ಈರುಳ್ಳಿ"],
    "potato": ["Potato", "ಆಲೂಗಡ್ಡೆ"],
    "banana": ["Banana", "ಬಾಳೆಹಣ್ಣು"],
    "tomato": ["Tomato", "ಟೊಮ್ಯಾಟೊ", "ಟೊಮೆಟೊ"],
}


class MarketService:
    """
    Service for querying, comparing, and formatting official Karnataka APMC market prices.
    """

    def __init__(self, client: Optional[BaseMarketClient] = None) -> None:
        """Initialize MarketService with given market client or default OfficialMarketClient."""
        self.client = client or OfficialMarketClient()

    def get_prices(
        self,
        crop: Optional[str] = None,
        location: Optional[LocationContext] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        date: Optional[str] = None,
    ) -> MarketContext:
        """
        Retrieves official APMC market prices for a specified crop and Karnataka location.

        Args:
            crop: Crop name in English or Kannada (e.g. 'ragi', 'ಮೆಕ್ಕೆಜೋಳ', 'tomato').
            location: Optional LocationContext object.
            district: Optional district name string.
            market: Optional specific market name string.
            date: Optional date string.

        Returns:
            MarketContext object containing verified price records or available=False.
        """
        canonical_crop = normalize_crop_name(crop) if crop else None
        if not canonical_crop:
            return MarketContext(
                available=False,
                query_crop=crop,
                status_message="Please specify an agricultural commodity or crop to check market prices."
            )

        # Resolve target district from location or explicit parameter
        target_district = ""
        if location and location.district:
            target_district = location.district.strip()
        elif district and district.strip():
            target_district = district.strip()

        # Query market client
        try:
            raw_records = self.client.fetch_market_prices(
                commodity=canonical_crop,
                district=target_district if target_district else None,
                market=market
            )
            # If district search returned no records, attempt broader state-level lookup for the crop
            if not raw_records and target_district:
                raw_records = self.client.fetch_market_prices(
                    commodity=canonical_crop,
                    district=None,
                    market=market
                )
        except MarketClientError as e:
            logger.warning("Market client error: %s", e)
            return MarketContext(
                available=False,
                query_crop=canonical_crop,
                query_district=target_district or None,
                query_market=market,
                status_message=f"Official market data could not be retrieved: {str(e)}"
            )
        except Exception as e:
            logger.error("Unexpected error fetching market data: %s", e)
            return MarketContext(
                available=False,
                query_crop=canonical_crop,
                query_district=target_district or None,
                query_market=market,
                status_message=f"Official market data is currently unavailable: {str(e)}"
            )

        if not raw_records:
            return MarketContext(
                available=False,
                query_crop=canonical_crop,
                query_district=target_district or None,
                query_market=market,
                status_message=f"No reported APMC market trading records found for '{canonical_crop}'."
            )

        # Parse records into structured MarketPriceRecord objects
        records: List[MarketPriceRecord] = []
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_dates: List[str] = []
        all_today = True

        for r in raw_records:
            rec_date = str(r.get("market_date", "")).strip()
            if rec_date:
                all_dates.append(rec_date)
                if rec_date != today_str:
                    all_today = False
            else:
                all_today = False

            try:
                min_p = float(r["min_price"]) if r.get("min_price") is not None else None
                max_p = float(r["max_price"]) if r.get("max_price") is not None else None
                mod_p = float(r["modal_price"]) if r.get("modal_price") is not None else None
                arr = float(r["arrivals"]) if r.get("arrivals") is not None else None
            except (ValueError, TypeError):
                min_p, max_p, mod_p, arr = None, None, None, None

            record = MarketPriceRecord(
                commodity=r.get("commodity", canonical_crop.capitalize()),
                canonical_crop=canonical_crop,
                market_name=r.get("market_name", r.get("market", "APMC Market")),
                district=r.get("district", target_district or "Karnataka"),
                state=r.get("state", "Karnataka"),
                market_date=rec_date,
                min_price=min_p,
                max_price=max_p,
                modal_price=mod_p,
                unit=r.get("unit", "₹/quintal"),
                arrivals=arr,
                variety=r.get("variety"),
                grade=r.get("grade"),
                source_authority=r.get("source_authority", "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB"),
                source_url=r.get("source_url", "https://agmarknet.gov.in")
            )
            records.append(record)

        latest_date = max(all_dates) if all_dates else today_str
        is_today = bool(all_today and records and all_dates)

        return MarketContext(
            available=True,
            query_crop=canonical_crop,
            query_district=target_district or None,
            query_market=market,
            is_today_data=is_today,
            latest_date=latest_date,
            records=records,
            source_authority=records[0].source_authority if records else "AGMARKNET & KSAMB",
            source_url=records[0].source_url if records else "https://agmarknet.gov.in",
            status_message="Success"
        )

    def format_market_context(self, context: MarketContext) -> str:
        """
        Formats MarketContext into a clean, factual, structured block for LLM prompt injection.
        """
        if not context or not context.available or not context.records:
            return ""

        lines: List[str] = [
            "--- OFFICIAL APMC MANDI MARKET PRICE CONTEXT ---"
        ]

        if context.is_today_data:
            lines.append(f"Trading Date: {context.latest_date} (Today's Reported Trading Session)")
        else:
            lines.append(f"Reporting Status: Latest available market data is dated {context.latest_date}")

        lines.append(f"Target Commodity: {context.query_crop.capitalize() if context.query_crop else 'Agricultural Commodity'}")
        lines.append("")
        lines.append("Reported APMC Market Prices:")

        for idx, rec in enumerate(context.records, 1):
            parts: List[str] = []
            if rec.market_name:
                parts.append(f"Market: {rec.market_name} ({rec.district})")
            if rec.market_date:
                parts.append(f"Date: {rec.market_date}")
            if rec.modal_price is not None:
                parts.append(f"Modal Price: {rec.modal_price} {rec.unit}")
            if rec.min_price is not None and rec.max_price is not None:
                parts.append(f"Range: {rec.min_price} - {rec.max_price} {rec.unit}")
            if rec.arrivals is not None:
                parts.append(f"Arrivals: {rec.arrivals} tonnes")
            if rec.variety:
                parts.append(f"Variety: {rec.variety}")

            lines.append(f"  {idx}. " + " | ".join(parts))

        lines.append("")
        lines.append(f"Source Authority: {context.source_authority} ({context.source_url})")
        lines.append("")
        lines.append("Important Farmer Price Guidance Rules:")
        lines.append("1. Reported prices reflect past official APMC transactions on the specified date. They are NOT guaranteed selling prices.")
        lines.append("2. Actual prices received depend on crop quality, moisture content, bag weight, daily market arrivals, and prevailing demand.")
        lines.append("3. Do NOT guarantee future profits or predict speculative price movements.")
        lines.append("4. Recommend farmers confirm prices with their local APMC secretary or electronic National Agriculture Market (e-NAM) portal before transporting produce.")

        return "\n".join(lines).strip()
