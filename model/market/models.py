"""
Data models representing Karnataka APMC Mandi commodity market prices for RaithaMitra.

Maintains strict truthfulness regarding trading dates, price ranges, units,
and anti-speculation disclaimers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class MarketPriceRecord:
    """
    Structured record representing commodity pricing at an official APMC / Mandi market.

    Attributes:
        commodity: Official reported commodity name (e.g., 'Ragi', 'Maize', 'Tomato').
        canonical_crop: Internal canonical crop name (e.g., 'ragi', 'maize', 'tomato').
        market_name: Name of the APMC market (e.g., 'Mandya', 'Belagavi', 'Binny Mill (F&V)').
        district: District where market is situated.
        state: State name (default: 'Karnataka').
        market_date: Date of market transactions (YYYY-MM-DD or DD/MM/YYYY as reported).
        min_price: Minimum reported price during the trading session.
        max_price: Maximum reported price during the trading session.
        modal_price: Modal / representative trading price if reported.
        unit: Price currency and measurement unit (e.g., '₹/quintal', '₹/kg').
        arrivals: Quantity of commodity arrivals reported (in tonnes or quintals if available).
        variety: Reported crop variety or classification if provided.
        grade: Quality grade (e.g. 'FAQ', 'Medium') if provided.
        source_authority: Official reporting authority (e.g., 'AGMARKNET / DMI, MoA&FW, GoI & KSAMB').
        source_url: Official portal URL.
        retrieved_at: ISO-8601 UTC timestamp of retrieval.
    """

    commodity: str
    canonical_crop: Optional[str] = None
    market_name: str = ""
    district: str = ""
    state: str = "Karnataka"
    market_date: str = ""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    unit: str = "₹/quintal"
    arrivals: Optional[float] = None
    variety: Optional[str] = None
    grade: Optional[str] = None
    source_authority: str = "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB"
    source_url: str = "https://agmarknet.gov.in"
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketPriceRecord:
        """Create MarketPriceRecord instance from dictionary representation."""
        return cls(
            commodity=data.get("commodity", ""),
            canonical_crop=data.get("canonical_crop"),
            market_name=data.get("market_name", ""),
            district=data.get("district", ""),
            state=data.get("state", "Karnataka"),
            market_date=data.get("market_date", ""),
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            modal_price=data.get("modal_price"),
            unit=data.get("unit", "₹/quintal"),
            arrivals=data.get("arrivals"),
            variety=data.get("variety"),
            grade=data.get("grade"),
            source_authority=data.get(
                "source_authority",
                "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            ),
            source_url=data.get("source_url", "https://agmarknet.gov.in"),
            retrieved_at=data.get(
                "retrieved_at", datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class MarketContext:
    """
    Structured container for APMC Mandi market context passed to advisory engines.

    Attributes:
        available: Boolean indicating whether authentic market data was retrieved.
        query_crop: Requested crop or commodity name.
        query_market: Specific market requested if any.
        query_district: Specific district queried if any.
        is_today_data: True ONLY if market_date matches today's trading date.
        latest_date: Date string of the latest available market record.
        records: List of individual MarketPriceRecord objects.
        source_authority: Reporting authority string.
        source_url: Official portal URL.
        retrieved_at: ISO-8601 UTC timestamp of context retrieval.
        status_message: Descriptive status or error message.
    """

    available: bool = True
    query_crop: Optional[str] = None
    query_market: Optional[str] = None
    query_district: Optional[str] = None
    is_today_data: bool = False
    latest_date: Optional[str] = None
    records: List[MarketPriceRecord] = field(default_factory=list)
    source_authority: str = "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB"
    source_url: str = "https://agmarknet.gov.in"
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status_message: str = "Success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to JSON-serializable dictionary."""
        d = asdict(self)
        d["records"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.records]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketContext:
        """Create MarketContext from dictionary representation."""
        raw_records = data.get("records", [])
        records = [
            MarketPriceRecord.from_dict(r) if isinstance(r, dict) else r
            for r in raw_records
        ]
        return cls(
            available=data.get("available", True),
            query_crop=data.get("query_crop"),
            query_market=data.get("query_market"),
            query_district=data.get("query_district"),
            is_today_data=data.get("is_today_data", False),
            latest_date=data.get("latest_date"),
            records=records,
            source_authority=data.get(
                "source_authority",
                "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB",
            ),
            source_url=data.get("source_url", "https://agmarknet.gov.in"),
            retrieved_at=data.get(
                "retrieved_at", datetime.now(timezone.utc).isoformat()
            ),
            status_message=data.get("status_message", "Success"),
        )
