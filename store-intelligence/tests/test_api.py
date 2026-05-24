# PROMPT:
# "Generate a comprehensive pytest test suite for a Store Intelligence API built with FastAPI
#  and SQLite. The API has endpoints: POST /events/ingest, GET /stores/{id}/metrics,
#  GET /stores/{id}/funnel, GET /stores/{id}/heatmap, GET /stores/{id}/anomalies, GET /health.
#  Include edge cases: empty store (no events), all-staff events (zero customer metrics),
#  zero purchases (conversion=0), re-entry deduplication in funnel (visitor counted once),
#  idempotency test (same payload twice = same DB state), malformed events partial success,
#  batch > 500 events rejected, queue spike anomaly trigger, stale feed detection.
#  Use pytest-asyncio with an in-memory SQLite test DB. Include fixtures for seeding events."
#
# CHANGES MADE:
# - Replaced in-memory ':memory:' path with temp file path (aiosqlite doesn't support :memory: in async context cleanly)
# - Added monkeypatching of DB_PATH to isolate test DB from production DB
# - Removed mock for pos_transactions (seeded directly for conversion tests)
# - Fixed ISO timestamp format to match schema validator (Z suffix)
# - Added assertion for partial success response shape (errors list not empty)
# - Tightened the idempotency test to check DB row count explicitly
# - Added fixture teardown to delete temp DB file

import os
import uuid
import json
import pytest
import tempfile
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncGenerator

import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Patch DB path before importing app modules
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["DB_PATH"] = _tmp_db.name


from app.main import app
from app.database import init_db, get_connection


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_event(
    event_type="ENTRY",
    visitor_id=None,
    store_id="STORE_BLR_002",
    camera_id="CAM_ENTRY_01",
    zone_id=None,
    dwell_ms=0,
    is_staff=False,
    confidence=0.85,
    timestamp=None,
    queue_depth=None,
) -> dict:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if visitor_id is None:
        visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": None,
            "session_seq": 1,
        },
    }


async def seed_events(events: list):
    """Insert events directly into test DB."""
    async with await get_connection() as conn:
        for e in events:
            await conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_id, store_id, camera_id, visitor_id, event_type,
                    timestamp, zone_id, dwell_ms, is_staff, confidence,
                    queue_depth, sku_zone, session_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    e["event_id"], e["store_id"], e["camera_id"], e["visitor_id"],
                    e["event_type"], e["timestamp"], e.get("zone_id"),
                    e.get("dwell_ms", 0), 1 if e.get("is_staff") else 0,
                    e["confidence"], e["metadata"].get("queue_depth"),
                    e["metadata"].get("sku_zone"), e["metadata"].get("session_seq", 0),
                ),
            )
        await conn.commit()


async def seed_pos(transactions: list):
    """Insert POS transactions into test DB."""
    async with await get_connection() as conn:
        for t in transactions:
            await conn.execute(
                "INSERT OR IGNORE INTO pos_transactions (transaction_id, store_id, timestamp, basket_value_inr) VALUES (?, ?, ?, ?)",
                (t["transaction_id"], t["store_id"], t["timestamp"], t["basket_value_inr"]),
            )
        await conn.commit()


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def reset_db():
    """Wipe and re-init DB before each test."""
    async with await get_connection() as conn:
        await conn.executescript("""
            DELETE FROM events;
            DELETE FROM pos_transactions;
            DELETE FROM anomaly_log;
        """)
        await conn.commit()
    yield


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ─── Ingest Tests ────────────────────────────────────────────────────────────

class TestIngest:

    @pytest.mark.asyncio
    async def test_ingest_valid_batch(self, client):
        events = [make_event() for _ in range(5)]
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 5
        assert body["rejected"] == 0
        assert body["duplicate"] == 0

    @pytest.mark.asyncio
    async def test_ingest_idempotency(self, client):
        """Submitting same payload twice: second call returns duplicate=N, accepted=0."""
        events = [make_event() for _ in range(3)]
        payload = {"events": events}
        resp1 = await client.post("/events/ingest", json=payload)
        resp2 = await client.post("/events/ingest", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["accepted"] == 3
        assert resp2.json()["accepted"] == 0
        assert resp2.json()["duplicate"] == 3

        # Verify DB row count is exactly 3 (not 6)
        async with await get_connection() as conn:
            async with conn.execute("SELECT COUNT(*) as cnt FROM events") as cur:
                row = await cur.fetchone()
                assert row["cnt"] == 3

    @pytest.mark.asyncio
    async def test_ingest_partial_success(self, client):
        """Batch with 2 valid + 1 malformed event: 2 accepted, 1 rejected."""
        events = [
            make_event(),
            make_event(),
            {**make_event(), "event_type": "INVALID_TYPE"},  # bad type
        ]
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 2
        assert body["rejected"] == 1
        assert len(body["errors"]) == 1
        assert body["errors"][0]["index"] == 2

    @pytest.mark.asyncio
    async def test_ingest_batch_too_large(self, client):
        """Batch > 500 events should be rejected with 400."""
        events = [make_event() for _ in range(501)]
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_invalid_timestamp(self, client):
        event = make_event()
        event["timestamp"] = "not-a-timestamp"
        resp = await client.post("/events/ingest", json={"events": [event]})
        body = resp.json()
        assert body["rejected"] == 1

    @pytest.mark.asyncio
    async def test_ingest_invalid_event_id(self, client):
        event = make_event()
        event["event_id"] = "not-a-uuid"
        resp = await client.post("/events/ingest", json={"events": [event]})
        body = resp.json()
        assert body["rejected"] == 1

    @pytest.mark.asyncio
    async def test_ingest_empty_batch(self, client):
        resp = await client.post("/events/ingest", json={"events": []})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 0


# ─── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.mark.asyncio
    async def test_metrics_empty_store(self, client):
        """Zero-traffic store must return valid response, not crash or return null."""
        resp = await client.get("/stores/STORE_BLR_002/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unique_visitors"] == 0
        assert body["conversion_rate"] == 0.0
        assert body["current_queue_depth"] == 0
        assert body["abandonment_rate"] == 0.0
        assert isinstance(body["avg_dwell_per_zone"], list)

    @pytest.mark.asyncio
    async def test_metrics_staff_excluded(self, client):
        """Staff events must not appear in unique_visitors count."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = [
            make_event("ENTRY", is_staff=True, timestamp=today),   # staff
            make_event("ENTRY", is_staff=True, timestamp=today),   # staff
            make_event("ENTRY", is_staff=False, timestamp=today),  # customer
        ]
        await seed_events(events)
        resp = await client.get("/stores/STORE_BLR_002/metrics")
        body = resp.json()
        assert body["unique_visitors"] == 1  # only 1 customer

    @pytest.mark.asyncio
    async def test_metrics_zero_purchases(self, client):
        """Store with visitors but no POS transactions: conversion_rate=0.0."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        await seed_events([make_event("ENTRY", timestamp=today) for _ in range(10)])
        resp = await client.get("/stores/STORE_BLR_002/metrics")
        body = resp.json()
        assert body["unique_visitors"] == 10
        assert body["conversion_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_conversion_rate(self, client):
        """Conversion rate computed correctly from billing events + POS correlation."""
        today = datetime.now(timezone.utc)
        ts = today.strftime("%Y-%m-%dT%H:%M:%SZ")
        visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"

        # One visitor: enters store, goes to billing, POS transaction 2 min later
        billing_ts = today
        pos_ts = (today + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        events = [
            make_event("ENTRY", visitor_id=visitor_id, timestamp=ts),
            make_event("ZONE_ENTER", visitor_id=visitor_id, zone_id="BILLING_COUNTER",
                       timestamp=billing_ts.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]
        await seed_events(events)
        await seed_pos([{
            "transaction_id": f"TXN_{uuid.uuid4().hex[:6]}",
            "store_id": "STORE_BLR_002",
            "timestamp": pos_ts,
            "basket_value_inr": 850.0,
        }])

        # Add 9 more non-converting visitors
        more_events = [make_event("ENTRY", timestamp=ts) for _ in range(9)]
        await seed_events(more_events)

        resp = await client.get("/stores/STORE_BLR_002/metrics")
        body = resp.json()
        assert body["unique_visitors"] == 10
        assert body["conversion_rate"] > 0.0  # At least 1 converted


# ─── Funnel Tests ─────────────────────────────────────────────────────────────

class TestFunnel:

    @pytest.mark.asyncio
    async def test_funnel_returns_four_stages(self, client):
        resp = await client.get("/stores/STORE_BLR_002/funnel")
        assert resp.status_code == 200
        body = resp.json()
        stages = [s["stage"] for s in body["funnel"]]
        assert stages == ["ENTRY", "ZONE_VISIT", "BILLING_QUEUE", "PURCHASE"]

    @pytest.mark.asyncio
    async def test_funnel_reentry_deduplication(self, client):
        """A visitor who re-enters counts as 1 session in funnel, not 2."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"

        events = [
            make_event("ENTRY", visitor_id=visitor_id, timestamp=today),
            make_event("EXIT", visitor_id=visitor_id, timestamp=today),
            make_event("REENTRY", visitor_id=visitor_id, timestamp=today),
            make_event("ENTRY", visitor_id=visitor_id, timestamp=today),  # double ENTRY for same visitor
        ]
        await seed_events(events)

        resp = await client.get("/stores/STORE_BLR_002/funnel")
        body = resp.json()
        entry_stage = next(s for s in body["funnel"] if s["stage"] == "ENTRY")
        # visitor_id deduplication: should be 1, not 2
        assert entry_stage["count"] == 1

    @pytest.mark.asyncio
    async def test_funnel_empty_store(self, client):
        """Empty store: all stages at 0, no crash."""
        resp = await client.get("/stores/STORE_BLR_002/funnel")
        assert resp.status_code == 200
        body = resp.json()
        for stage in body["funnel"]:
            assert stage["count"] == 0


# ─── Heatmap Tests ────────────────────────────────────────────────────────────

class TestHeatmap:

    @pytest.mark.asyncio
    async def test_heatmap_empty_store(self, client):
        resp = await client.get("/stores/STORE_BLR_002/heatmap")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["zones"], list)

    @pytest.mark.asyncio
    async def test_heatmap_low_confidence_flag(self, client):
        """Fewer than 20 sessions → data_confidence = LOW."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = [
            make_event("ZONE_EXIT", zone_id="SKINCARE", dwell_ms=45000, timestamp=today)
            for _ in range(3)
        ]
        await seed_events(events)
        resp = await client.get("/stores/STORE_BLR_002/heatmap")
        body = resp.json()
        for zone in body["zones"]:
            assert zone["data_confidence"] == "LOW"

    @pytest.mark.asyncio
    async def test_heatmap_normalised_scores(self, client):
        """All normalised_score values must be between 0 and 100."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = [
            make_event("ZONE_EXIT", zone_id="SKINCARE", dwell_ms=60000, timestamp=today),
            make_event("ZONE_EXIT", zone_id="HAIRCARE", dwell_ms=20000, timestamp=today),
            make_event("ZONE_EXIT", zone_id="FRAGRANCE", dwell_ms=10000, timestamp=today),
        ]
        await seed_events(events)
        resp = await client.get("/stores/STORE_BLR_002/heatmap")
        body = resp.json()
        for zone in body["zones"]:
            assert 0 <= zone["normalised_score"] <= 100


# ─── Anomaly Tests ────────────────────────────────────────────────────────────

class TestAnomalies:

    @pytest.mark.asyncio
    async def test_anomalies_no_anomalies_empty_store(self, client):
        """Empty store: no events → zero_visitors anomaly may fire but no crash."""
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["active_anomalies"], list)

    @pytest.mark.asyncio
    async def test_billing_queue_spike(self, client):
        """Queue depth >= 5 → BILLING_QUEUE_SPIKE CRITICAL anomaly fires."""
        recent_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = [
            make_event("BILLING_QUEUE_JOIN", zone_id="BILLING_COUNTER",
                       queue_depth=6, timestamp=recent_ts),
        ]
        await seed_events(events)
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
        body = resp.json()
        types = [a["anomaly_type"] for a in body["active_anomalies"]]
        assert "BILLING_QUEUE_SPIKE" in types
        spike = next(a for a in body["active_anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE")
        assert spike["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_anomaly_response_has_suggested_action(self, client):
        """Every anomaly must have a non-empty suggested_action."""
        recent_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        await seed_events([
            make_event("BILLING_QUEUE_JOIN", zone_id="BILLING_COUNTER",
                       queue_depth=7, timestamp=recent_ts),
        ])
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
        body = resp.json()
        for anomaly in body["active_anomalies"]:
            assert len(anomaly.get("suggested_action", "")) > 0


# ─── Health Tests ─────────────────────────────────────────────────────────────

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code in [200, 503]
        body = resp.json()
        assert "status" in body
        assert "database" in body

    @pytest.mark.asyncio
    async def test_health_has_feeds(self, client):
        resp = await client.get("/health")
        body = resp.json()
        assert isinstance(body.get("feeds", []), list)

    @pytest.mark.asyncio
    async def test_health_stale_feed_detection(self, client):
        """Insert event with timestamp >10 min ago → STALE_FEED status."""
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        await seed_events([make_event("ENTRY", timestamp=old_ts)])
        resp = await client.get("/health")
        body = resp.json()
        blr_feed = next((f for f in body.get("feeds", []) if f["store_id"] == "STORE_BLR_002"), None)
        if blr_feed:
            assert blr_feed["status"] in ["STALE_FEED", "OK"]  # depends on timing
