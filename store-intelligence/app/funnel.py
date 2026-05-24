"""
funnel.py — Conversion funnel endpoint.

GET /stores/{store_id}/funnel

Funnel stages (session is the unit, not raw events):
  1. Entry          — unique visitor sessions (non-staff ENTRY events)
  2. Zone Visit     — sessions that visited at least one product zone
  3. Billing Queue  — sessions that entered BILLING_COUNTER
  4. Purchase       — sessions correlated with a POS transaction

Re-entries: A visitor with a REENTRY event is counted ONCE per day's sessions.
The funnel is de-duplicated at the visitor_id level.

Key design: we use visitor_id as the session unit, not raw event counts.
This prevents re-entry inflation and cross-camera double-counting.
"""

import logging
from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models import FunnelResponse, FunnelStage
from database import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_store_funnel(store_id: str, request: Request):
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    Session is the unit. Re-entries do not inflate visitor counts.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    today = date.today().isoformat()
    start_ts = f"{today}T00:00:00Z"
    end_ts = f"{today}T23:59:59Z"

    try:
        async with await get_connection() as conn:

            # Stage 1: Unique customer sessions (ENTRY, non-staff, deduplicated by visitor_id)
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as entry_sessions
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ENTRY'
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                entry_count = row["entry_sessions"] if row else 0

            # Stage 2: Sessions that visited a product zone
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as zone_visitors
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ZONE_ENTER'
                  AND zone_id NOT IN ('ENTRY_THRESHOLD', 'BILLING_COUNTER', 'STOCKROOM', 'BEAUTY_ADVISOR_DESK')
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                  AND visitor_id IN (
                      SELECT DISTINCT visitor_id FROM events
                      WHERE store_id = ? AND event_type = 'ENTRY'
                        AND is_staff = 0 AND timestamp BETWEEN ? AND ?
                  )
                """,
                (store_id, start_ts, end_ts, store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                zone_count = row["zone_visitors"] if row else 0

            # Stage 3: Sessions that entered billing zone
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as billing_visitors
                FROM events
                WHERE store_id = ?
                  AND zone_id = 'BILLING_COUNTER'
                  AND event_type = 'ZONE_ENTER'
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                  AND visitor_id IN (
                      SELECT DISTINCT visitor_id FROM events
                      WHERE store_id = ? AND event_type = 'ENTRY'
                        AND is_staff = 0 AND timestamp BETWEEN ? AND ?
                  )
                """,
                (store_id, start_ts, end_ts, store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                billing_count = row["billing_visitors"] if row else 0

            # Stage 4: Sessions correlated with a POS transaction
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT e.visitor_id) as purchase_sessions
                FROM events e
                INNER JOIN pos_transactions p
                    ON p.store_id = e.store_id
                    AND datetime(p.timestamp) BETWEEN
                        datetime(e.timestamp, '-5 minutes') AND
                        datetime(e.timestamp, '+1 minutes')
                WHERE e.store_id = ?
                  AND e.zone_id = 'BILLING_COUNTER'
                  AND e.is_staff = 0
                  AND e.timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                row = await cur.fetchone()
                purchase_count = row["purchase_sessions"] if row else 0

    except Exception as e:
        logger.error(f'"Funnel query failed store_id={store_id}: {e}"')
        return JSONResponse(
            status_code=503,
            content={
                "error": "DATABASE_UNAVAILABLE",
                "message": "Funnel data temporarily unavailable.",
                "trace_id": trace_id,
            },
        )

    def drop_pct(current: int, previous: int) -> float:
        if previous == 0:
            return 0.0
        return round((1 - current / previous) * 100, 1)

    funnel_stages = [
        FunnelStage(stage="ENTRY", count=entry_count, drop_off_pct=0.0),
        FunnelStage(stage="ZONE_VISIT", count=zone_count, drop_off_pct=drop_pct(zone_count, entry_count)),
        FunnelStage(stage="BILLING_QUEUE", count=billing_count, drop_off_pct=drop_pct(billing_count, zone_count)),
        FunnelStage(stage="PURCHASE", count=purchase_count, drop_off_pct=drop_pct(purchase_count, billing_count)),
    ]

    return FunnelResponse(
        store_id=store_id,
        funnel=funnel_stages,
        session_window=f"{today}T00:00:00Z / {today}T23:59:59Z",
    )
