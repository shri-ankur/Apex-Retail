"""
heatmap.py — Zone visit frequency + dwell heatmap.

GET /stores/{store_id}/heatmap

Returns zone-level visit frequency and average dwell normalised 0-100.
data_confidence is LOW if fewer than 20 sessions in the window.
"""

import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models import HeatmapResponse, ZoneHeatmapEntry
from database import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_store_heatmap(store_id: str, request: Request):
    trace_id = getattr(request.state, "trace_id", "unknown")
    today = date.today().isoformat()
    start_ts = f"{today}T00:00:00Z"
    end_ts = f"{today}T23:59:59Z"

    try:
        async with await get_connection() as conn:
            # Zone visit frequency + avg dwell (from ZONE_EXIT events with dwell_ms)
            async with conn.execute(
                """
                SELECT
                    zone_id,
                    COUNT(DISTINCT visitor_id) as visit_frequency,
                    AVG(CASE WHEN dwell_ms > 0 THEN dwell_ms ELSE NULL END) / 1000.0 as avg_dwell_s
                FROM events
                WHERE store_id = ?
                  AND event_type IN ('ZONE_EXIT', 'ZONE_DWELL')
                  AND zone_id IS NOT NULL
                  AND zone_id NOT IN ('ENTRY_THRESHOLD', 'STOCKROOM')
                  AND is_staff = 0
                  AND timestamp BETWEEN ? AND ?
                GROUP BY zone_id
                ORDER BY visit_frequency DESC
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                rows = await cur.fetchall()

            # Total sessions for confidence flag
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as total
                FROM events
                WHERE store_id = ? AND event_type = 'ENTRY'
                  AND is_staff = 0 AND timestamp BETWEEN ? AND ?
                """,
                (store_id, start_ts, end_ts),
            ) as cur:
                total_row = await cur.fetchone()
                total_sessions = total_row["total"] if total_row else 0

    except Exception as e:
        logger.error(f'"Heatmap query failed store_id={store_id}: {e}"')
        return JSONResponse(
            status_code=503,
            content={"error": "DATABASE_UNAVAILABLE", "message": "Heatmap unavailable.", "trace_id": trace_id},
        )

    if not rows:
        return HeatmapResponse(
            store_id=store_id,
            zones=[],
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # Normalise visit_frequency 0-100
    max_freq = max(r["visit_frequency"] for r in rows) or 1
    max_dwell = max((r["avg_dwell_s"] or 0) for r in rows) or 1

    zones = []
    for r in rows:
        freq = r["visit_frequency"]
        dwell = r["avg_dwell_s"] or 0.0
        # Combined score: 60% frequency weight + 40% dwell weight
        normalised = round(((freq / max_freq) * 60 + (dwell / max_dwell) * 40), 1)
        confidence = "LOW" if total_sessions < 20 else "HIGH"

        zones.append(
            ZoneHeatmapEntry(
                zone_id=r["zone_id"],
                visit_frequency=freq,
                avg_dwell_seconds=round(dwell, 1),
                normalised_score=min(normalised, 100.0),
                data_confidence=confidence,
            )
        )

    return HeatmapResponse(
        store_id=store_id,
        zones=zones,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
