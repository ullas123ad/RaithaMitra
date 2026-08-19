"""
Unit tests for the RaithaMitra Karnataka APMC Mandi Market Price Module.
Validates crop resolution, location integration, date truthfulness, anti-fabrication,
failure behavior, unit consistency, and pipeline integration.
"""

import time
import unittest
from typing import Dict, Any

from model.location.service import LocationService
from model.market.models import MarketPriceRecord, MarketContext
from model.market.client import MockMarketClient, MarketClientError
from model.market.service import MarketService
from model.advisory.agriparam_engine import (
    AdvisoryEngine,
    AdvisoryConfig,
    MockAdvisoryBackend,
)
from model.advisory.language_bridge import MockLanguageBridge


class TestMarketServiceCore(unittest.TestCase):
    """Validates core MarketService operations with MockMarketClient."""

    def setUp(self) -> None:
        self.mock_client = MockMarketClient()
        self.service = MarketService(client=self.mock_client)
        self.loc_service = LocationService()

    def test_1_mandya_ragi_retrieval(self) -> None:
        """TEST 1: Verify Mandya APMC ragi price retrieval and schema."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        ctx = self.service.get_prices(crop="ragi", location=loc)

        self.assertTrue(ctx.available)
        self.assertEqual(ctx.query_crop, "ragi")
        self.assertEqual(ctx.query_district, "Mandya")
        self.assertEqual(len(ctx.records), 1)

        rec = ctx.records[0]
        self.assertEqual(rec.market_name, "Mandya")
        self.assertEqual(rec.min_price, 2800.0)
        self.assertEqual(rec.max_price, 3400.0)
        self.assertEqual(rec.modal_price, 3200.0)
        self.assertEqual(rec.unit, "₹/quintal")
        self.assertEqual(rec.arrivals, 45.0)
        self.assertEqual(rec.market_date, "2026-08-19")
        self.assertIn("AGMARKNET", rec.source_authority)

    def test_2_maize_belagavi_date_handling(self) -> None:
        """TEST 2: Verify Belagavi maize price preserves historical date and does not claim today."""
        loc = self.loc_service.get_location(district="Belagavi", taluk="Gokak", village="Arbhavi")
        ctx = self.service.get_prices(crop="maize", location=loc)

        self.assertTrue(ctx.available)
        self.assertEqual(ctx.query_crop, "maize")
        self.assertEqual(len(ctx.records), 1)

        rec = ctx.records[0]
        self.assertEqual(rec.market_name, "Belagavi")
        self.assertEqual(rec.market_date, "2026-08-18")
        self.assertEqual(rec.modal_price, 2350.0)

        # Formatted string must explicitly state date and not claim today
        formatted = self.service.format_market_context(ctx)
        self.assertIn("2026-08-18", formatted)
        self.assertIn("Latest available market data is dated 2026-08-18", formatted)

    def test_3_tomato_bengaluru_retrieval(self) -> None:
        """TEST 3: Verify Bengaluru Binny Mill tomato price retrieval."""
        loc = self.loc_service.get_location(district="Bengaluru Urban")
        ctx = self.service.get_prices(crop="tomato", location=loc)

        self.assertTrue(ctx.available)
        self.assertEqual(len(ctx.records), 1)
        rec = ctx.records[0]
        self.assertEqual(rec.market_name, "Binny Mill (F&V)")
        self.assertEqual(rec.modal_price, 1800.0)
        self.assertEqual(rec.unit, "₹/quintal")

    def test_4_onion_multi_market_comparison(self) -> None:
        """TEST 4: Verify onion price query across Karnataka returns multiple APMC records."""
        ctx = self.service.get_prices(crop="onion")

        self.assertTrue(ctx.available)
        self.assertGreaterEqual(len(ctx.records), 2)
        markets = [r.market_name for r in ctx.records]
        self.assertIn("Hubballi (Amaragol)", markets)
        self.assertIn("Yeshwanthpur", markets)

    def test_5_english_crop_input(self) -> None:
        """TEST 5: Verify English crop name resolution."""
        ctx = self.service.get_prices(crop="Finger Millet", district="Mandya")
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.query_crop, "ragi")

    def test_6_kannada_crop_aliases(self) -> None:
        """TEST 6: Verify Kannada crop aliases resolve to canonical identity."""
        for alias in ["ರಾಗಿ", "ಮೆಕ್ಕೆಜೋಳ", "ಟೊಮ್ಯಾಟೊ", "ಈರುಳ್ಳಿ"]:
            ctx = self.service.get_prices(crop=alias)
            self.assertTrue(ctx.available)
            self.assertIsNotNone(ctx.query_crop)

    def test_7_market_unavailable_failure_tolerance(self) -> None:
        """TEST 7: Verify provider failure returns available=False safely without throwing unhandled error."""
        failing_client = MockMarketClient(should_fail=True)
        svc = MarketService(client=failing_client)

        ctx = svc.get_prices(crop="ragi", district="Mandya")
        self.assertFalse(ctx.available)
        self.assertEqual(len(ctx.records), 0)
        self.assertTrue(
            "could not be retrieved" in ctx.status_message.lower() or
            "unavailable" in ctx.status_message.lower()
        )

        # Formatter returns empty string safely
        self.assertEqual(svc.format_market_context(ctx), "")

    def test_8_no_crop_specified(self) -> None:
        """TEST 8: Verify query with missing crop asks for commodity and does not guess."""
        ctx = self.service.get_prices(crop=None)
        self.assertFalse(ctx.available)
        self.assertIn("Please specify an agricultural commodity", ctx.status_message)

    def test_9_non_agricultural_crop(self) -> None:
        """TEST 9: Verify non-agricultural term is rejected gracefully."""
        ctx = self.service.get_prices(crop="laptop")
        self.assertFalse(ctx.available)

    def test_10_old_data_protection(self) -> None:
        """TEST 10: Verify older records set is_today_data=False and format dated notice."""
        historical_records = [
            {
                "commodity": "Paddy",
                "canonical_crop": "paddy",
                "market_name": "Mysuru",
                "district": "Mysuru",
                "state": "Karnataka",
                "market_date": "2026-08-10",  # 9 days ago
                "min_price": 2200.0,
                "max_price": 2600.0,
                "modal_price": 2400.0,
                "unit": "₹/quintal",
            }
        ]
        svc = MarketService(client=MockMarketClient(records=historical_records))
        ctx = svc.get_prices(crop="paddy", district="Mysuru")

        self.assertTrue(ctx.available)
        self.assertFalse(ctx.is_today_data)
        self.assertEqual(ctx.latest_date, "2026-08-10")

        text = svc.format_market_context(ctx)
        self.assertIn("Latest available market data is dated 2026-08-10", text)
        self.assertNotIn("Today's Reported Trading Session", text)


class TestAdvisoryEngineMarketIntegration(unittest.TestCase):
    """Validates end-to-end AdvisoryEngine integration with MarketContext."""

    def setUp(self) -> None:
        self.mock_market = MockMarketClient()
        self.engine = AdvisoryEngine(
            config=AdvisoryConfig(backend="mock", use_rag=True),
            backend=MockAdvisoryBackend(),
            language_bridge=MockLanguageBridge(),
            market_service=MarketService(client=self.mock_market)
        )
        self.loc_service = LocationService()

    def test_11_advisory_engine_attaches_market_context(self) -> None:
        """TEST 11: Verify AdvisoryEngine attaches market context for price queries."""
        loc = self.loc_service.get_location(district="Mandya", taluk="Pandavapura", village="Melukote")
        res = self.engine.generate_advisory(
            query="ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
            source_language="kn",
            location=loc,
            crop="ragi"
        )
        self.assertIn("market", res)
        self.assertIsNotNone(res["market"])
        self.assertTrue(res["market"]["available"])
        self.assertEqual(res["market"]["query_crop"], "ragi")
        self.assertTrue(len(res["market"]["records"]) > 0)
        self.assertEqual(res["market"]["records"][0]["modal_price"], 3200.0)

    def test_12_non_market_query_leaves_market_clean(self) -> None:
        """TEST 12: Verify non-price query does not pollute response."""
        res = self.engine.generate_advisory(
            query="PM-KISAN ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು.",
            source_language="kn"
        )
        self.assertIn("market", res)
        self.assertIsNone(res["market"])

    def test_13_performance_benchmark(self) -> None:
        """TEST 13: Verify MarketService init (<20ms) and mock lookup (<5ms)."""
        t0 = time.perf_counter()
        svc = MarketService(client=MockMarketClient())
        init_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(100):
            svc.get_prices(crop="ragi", district="Mandya")
        avg_lookup_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

        self.assertLess(init_ms, 20.0, f"MarketService init took {init_ms:.2f} ms")
        self.assertLess(avg_lookup_ms, 5.0, f"Average market lookup took {avg_lookup_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
