"""
anomalies.py — Real-time anomaly detection.

GET /stores/{store_id}/anomalies

Detected anomaly types:
  - BILLING_QUEUE_SPIKE   CRITICAL  Queue depth > 5 or 2x 7-day average
  - CONVERSION_DROP       WARN      Today's conversion rate < 70% of 7-day avg
  - DEAD_ZONE             WARN      No ZONE_ENTER events in any zone for 30+ minutes
  - STALE_FEED            WARN      No events received for >10 minutes
  - ZERO_VISITORS         INFO      Zero customer entries in last 30 minutes (outside open hours = expected)
"""

import logging
import uuid
from datetime import date, datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models import AnomalyResponse, Anomaly
from database import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ts_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/stores/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_store_anomalies(store_id: str, request: Request):
    trace_id = getattr(request.state, "trace_id", "unknown")
    now = now_utc()
    today = date.today().isoformat()
    start_today = f"{today}T00:00:00Z"
    end_today = f"{today}T23:59:59Z"
    thirty_min_ago = ts_str(now - timedelta(minutes=30))
    ten_min_ago = ts_str(now - timedelta(minutes=10))
    seven_days_ago = ts_str(now - timedelta(days=7))

    anomalies: List[Anomaly] = []

    try:
        async with await get_connection() as conn:

            # --- Anomaly 1: BILLING_QUEUE_SPIKE ---
            async with conn.execute(
                """
                SELECT MAX(queue_depth) as max_q
                FROM events
                WHERE store_id = ?
                  AND event_type = 'BILLING_QUEUE_JOIN'
                  AND timestamp >= ?
                """,
                (store_id, thirty_min_ago),
            ) as cur:
                row = await cur.fetchone()
                current_max_q = row["max_q"] or 0

            async with conn.execute(
                """
                SELECT AVG(queue_depth) as avg_q
                FROM events
                WHERE store_id = ?
                  AND event_type = 'BILLING_QUEUE_JOIN'
                  AND timestamp BETWEEN ? AND ?
                  AND timestamp < ?
                """,
                (store_id, seven_days_ago, start_today, thirty_min_ago),
            ) as cur:
                row = await cur.fetchone()
                avg_q_7d = row["avg_q"] or 0

            if current_max_q >= 5 or (avg_q_7d > 0 and current_max_q >= avg_q_7d * 2):
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type="BILLING_QUEUE_SPIKE",
                    severity="CRITICAL",
                    description=f"Billing queue depth at {current_max_q} (7-day avg: {avg_q_7d:.1f}). Queue spike detected.",
                    suggested_action="Open additional billing counter immediately. Alert floor manager.",
                    detected_at=ts_str(now),
                    metadata={"current_queue_depth": current_max_q, "avg_7d": round(avg_q_7d, 1)},
                ))

            # --- Anomaly 2: CONVERSION_DROP ---
            async with conn.execute(
                """
                SELECT
                    COUNT(DISTINCT visitor_id) as total_visitors,
                    (SELECT COUNT(DISTINCT e2.visitor_id)
                     FROM events e2
                     INNER JOIN pos_transactions p ON p.store_id = e2.store_id
                       AND datetime(p.timestamp) BETWEEN
                           datetime(e2.timestamp, '-5 minutes') AND datetime(e2.timestamp, '+1 minutes')
                     WHERE e2.store_id = ? AND e2.zone_id = 'BILLING_COUNTER'
                       AND e2.is_staff = 0 AND e2.timestamp BETWEEN ? AND ?
                    ) as converted
                FROM events
                WHERE store_id = ? AND event_type = 'ENTRY'
                  AND is_staff = 0 AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_today, end_today, store_id, start_today, end_today),
            ) as cur:
                row = await cur.fetchone()
                today_visitors = row["total_visitors"] or 0
                today_converted = row["converted"] or 0
                today_conversion = (today_converted / today_visitors) if today_visitors > 0 else 0.0

            async with conn.execute(
                """
                SELECT
                    COUNT(DISTINCT visitor_id) as hist_visitors,
                    COUNT(DISTINCT CASE WHEN zone_id = 'BILLING_COUNTER' THEN visitor_id END) as hist_billing
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ENTRY'
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                """,
                (store_id, seven_days_ago, start_today),
            ) as cur:
                row = await cur.fetchone()
                hist_visitors = row["hist_visitors"] or 0
                hist_billing = row["hist_billing"] or 0
                avg_conversion_7d = (hist_billing / hist_visitors) if hist_visitors > 0 else 0.0

            if avg_conversion_7d > 0 and today_conversion < avg_conversion_7d * 0.7 and today_visitors >= 10:
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type="CONVERSION_DROP",
                    severity="WARN",
                    description=f"Today's conversion rate {today_conversion:.1%} is below 70% of 7-day average ({avg_conversion_7d:.1%}).",
                    suggested_action="Review zone heatmap for engagement drop. Check if promotions are visible. Consider floor staff redeployment.",
                    detected_at=ts_str(now),
                    metadata={
                        "today_conversion": round(today_conversion, 4),
                        "avg_7d_conversion": round(avg_conversion_7d, 4),
                        "today_visitors": today_visitors,
                    },
                ))

            # --- Anomaly 3: DEAD_ZONE ---
            async with conn.execute(
                """
                SELECT zone_id, MAX(timestamp) as last_visit
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ZONE_ENTER'
                  AND is_staff = 0
                  AND zone_id NOT IN ('ENTRY_THRESHOLD', 'BILLING_COUNTER', 'STOCKROOM')
                GROUP BY zone_id
                """,
                (store_id,),
            ) as cur:
                zone_rows = await cur.fetchall()

            for zrow in zone_rows:
                if zrow["last_visit"] and zrow["last_visit"] < thirty_min_ago:
                    anomalies.append(Anomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="DEAD_ZONE",
                        severity="WARN",
                        description=f"Zone '{zrow['zone_id']}' has had no customer visits in over 30 minutes.",
                        suggested_action=f"Check camera feed for '{zrow['zone_id']}'. If feed is live, consider moving staff to guide customers to this zone.",
                        detected_at=ts_str(now),
                        metadata={"zone_id": zrow["zone_id"], "last_visit_at": zrow["last_visit"]},
                    ))

            # --- Anomaly 4: STALE_FEED ---
            async with conn.execute(
                """
                SELECT MAX(timestamp) as last_ts FROM events WHERE store_id = ?
                """,
                (store_id,),
            ) as cur:
                row = await cur.fetchone()
                last_ts = row["last_ts"] if row and row["last_ts"] else None

            if last_ts and last_ts < ten_min_ago:
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type="STALE_FEED",
                    severity="WARN",
                    description=f"No events received from store in over 10 minutes. Last event: {last_ts}",
                    suggested_action="Check pipeline health. Verify camera connectivity and detection service is running.",
                    detected_at=ts_str(now),
                    metadata={"last_event_at": last_ts},
                ))

            # --- Anomaly 5: ZERO_VISITORS ---
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as recent_visitors
                FROM events
                WHERE store_id = ?
                  AND event_type = 'ENTRY'
                  AND is_staff = 0
                  AND timestamp >= ?
                """,
                (store_id, thirty_min_ago),
            ) as cur:
                row = await cur.fetchone()
                recent_visitors = row["recent_visitors"] or 0

            if recent_visitors == 0 and last_ts:  # Only flag if we have a live feed
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type="ZERO_VISITORS",
                    severity="INFO",
                    description="No customer entries detected in the last 30 minutes.",
                    suggested_action="Verify this is expected (store closed / off-hours). If during business hours, check entry camera.",
                    detected_at=ts_str(now),
                    metadata={"window_minutes": 30},
                ))

    except Exception as e:
        logger.error(f'"Anomaly detection failed store_id={store_id}: {e}"')
        return JSONResponse(
            status_code=503,
            content={"error": "DATABASE_UNAVAILABLE", "message": "Anomaly check unavailable.", "trace_id": trace_id},
        )

    return AnomalyResponse(
        store_id=store_id,
        active_anomalies=anomalies,
        checked_at=ts_str(now),
    )
