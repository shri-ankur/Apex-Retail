"""
metrics.py — Real-time store metrics computation.

GET /stores/{store_id}/metrics

Computes on every request (not cached) to ensure real-time accuracy:
  - unique_visitors: distinct visitor_id count with ENTRY events today (staff excluded)
  - conversion_rate: visitors who completed a purchase / total unique visitors
  - avg_dwell_per_zone: mean dwell per zone from ZONE_EXIT + ZONE_DWELL events
  - current_queue_depth: latest queue_depth from BILLING_QUEUE_JOIN
  - abandonment_rate: BILLING_QUEUE_ABANDON / BILLING_QUEUE_JOIN
  - total_revenue_inr: sum from pos_transactions for today

Conversion correlation:
  A visitor who was in the BILLING_COUNTER zone within 5 minutes before a POS
  transaction timestamp is counted as converted (per spec §3.4).
"""

import logging
from datetime import datetime, timezone, date
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from models import StoreMetricsResponse, ZoneDwellMetric
from database import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


def today_range() -> tuple[str, str]:
    """Return ISO timestamps for start and end of today (UTC)."""
    today = date.today().isoformat()
    return f"{today}T00:00:00Z", f"{today}T23:59:59Z"


@router.get("/stores/{store_id}/metrics", response_model=StoreMetricsResponse)
async def get_store_metrics(store_id: str, request: Request):
    """
    Real-time metrics for a store today.
    Staff events (is_staff=1) are excluded from all customer metrics.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    start_ts, end_ts = today_range()

    try:
        async with await get_connection() as conn:

            # 1. Unique visitors today (non-staff ENTRY events)
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as cnt
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ENTRY'
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                unique_visitors = row["cnt"] if row else 0

            # 2. Converted visitors: billing zone visit correlated to POS transaction (5-min window)
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT e.visitor_id) as converted
                FROM events e
                INNER JOIN pos_transactions p
                    ON p.store_id = e.store_id
                    AND datetime(p.timestamp) BETWEEN datetime(e.timestamp, '-5 minutes')
                                                  AND datetime(e.timestamp, '+1 minutes')
                WHERE e.store_id = ?
                  AND e.zone_id = 'BILLING_COUNTER'
                  AND e.is_staff = 0
                  AND e.timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                converted = row["converted"] if row else 0

            conversion_rate = round(converted / unique_visitors, 4) if unique_visitors > 0 else 0.0

            # 3. Average dwell per zone (from ZONE_EXIT dwell_ms, staff excluded)
            async with conn.execute(
                """
                SELECT zone_id,
                       AVG(dwell_ms) / 1000.0 as avg_dwell_s,
                       COUNT(DISTINCT visitor_id) as visitor_count
                FROM events
                WHERE store_id = ?
                  AND event_type IN ('ZONE_EXIT', 'ZONE_DWELL')
                  AND zone_id IS NOT NULL
                  AND is_staff = 0
                  AND dwell_ms > 0
                  AND timestamp BETWEEN ? AND ?
                GROUP BY zone_id
                ORDER BY avg_dwell_s DESC
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                zone_rows = await cur.fetchall()

            avg_dwell_per_zone = [
                ZoneDwellMetric(
                    zone_id=r["zone_id"],
                    avg_dwell_seconds=round(r["avg_dwell_s"], 1),
                    visitor_count=r["visitor_count"],
                )
                for r in zone_rows
                if r["zone_id"] is not None
            ]

            # 4. Current queue depth (latest BILLING_QUEUE_JOIN queue_depth value)
            async with conn.execute(
                """
                SELECT queue_depth
                FROM events
                WHERE store_id = ?
                  AND event_type = 'BILLING_QUEUE_JOIN'
                  AND queue_depth IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (store_id,),
            ) as cur:
                row = await cur.fetchone()
                current_queue_depth = row["queue_depth"] if row else 0

            # 5. Abandonment rate
            async with conn.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type = 'BILLING_QUEUE_JOIN' THEN 1 ELSE 0 END) AS joins,
                    SUM(CASE WHEN event_type = 'BILLING_QUEUE_ABANDON' THEN 1 ELSE 0 END) AS abandons
                FROM events
                WHERE store_id = ?
                  AND event_type IN ('BILLING_QUEUE_JOIN', 'BILLING_QUEUE_ABANDON')
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                joins = row["joins"] or 0
                abandons = row["abandons"] or 0
                abandonment_rate = round(abandons / joins, 4) if joins > 0 else 0.0

            # 6. Total revenue today
            async with conn.execute(
                """
                SELECT COALESCE(SUM(basket_value_inr), 0.0) as revenue
                FROM pos_transactions
                WHERE store_id = ? AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                total_revenue = float(row["revenue"]) if row else 0.0

            # 7. Last event timestamp for as_of field
            async with conn.execute(
                """
                SELECT MAX(timestamp) as last_ts FROM events WHERE store_id = ?
                """,
                (store_id,),
            ) as cur:
                row = await cur.fetchone()
                as_of = row["last_ts"] if row and row["last_ts"] else "N/A"

    except Exception as e:
        logger.error(f'"Metrics query failed store_id={store_id}: {e}"')
        return JSONResponse(
            status_code=503,
            content={
                "error": "DATABASE_UNAVAILABLE",
                "message": "Metrics temporarily unavailable. Please retry.",
                "trace_id": trace_id,
            },
        )

    return StoreMetricsResponse(
        store_id=store_id,
        date=date.today().isoformat(),
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_per_zone=avg_dwell_per_zone,
        current_queue_depth=current_queue_depth,
        abandonment_rate=abandonment_rate,
        total_revenue_inr=total_revenue,
        as_of=as_of,
    )
