"""
health.py — Service health endpoint.

GET /health

Returns:
  - Service status (OK / DEGRADED)
  - Database status
  - Per-store feed status with STALE_FEED warning if lag > 10 minutes
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models import HealthResponse, StoreFeedStatus
from database import get_connection, get_db_health

logger = logging.getLogger(__name__)
router = APIRouter()

KNOWN_STORES = ["STORE_BLR_002", "STORE_BLR_001", "STORE_MUM_001", "STORE_DEL_001", "STORE_HYD_001"]

# All 5 cameras mapped to this store (from frame inspection)
STORE_CAMERAS = {
    "STORE_BLR_002": [
        "CAM_ENTRY_01",      # CAM_3.mp4  glass door threshold
        "CAM_FLOOR_A",       # CAM_1.mp4  skincare wall
        "CAM_FLOOR_B",       # CAM_2.mp4  makeup wall
        "CAM_BILLING_01",    # CAM_5.mp4  POS counter
        "CAM_STOCKROOM_01",  # CAM_4.mp4  back stockroom (staff-only)
    ]
}
STALE_THRESHOLD_MINUTES = 10


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check. Accurate — this is what an on-call engineer checks first.
    """
    now = now_utc()
    checked_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Check database
    db_health = await get_db_health()
    db_status = db_health["status"]

    if db_status != "OK":
        return JSONResponse(
            status_code=503,
            content={
                "status": "DEGRADED",
                "database": "UNAVAILABLE",
                "error": db_health.get("error", "Unknown"),
                "feeds": [],
                "checked_at": checked_at,
            },
        )

    # Check per-store feed freshness
    feeds: List[StoreFeedStatus] = []
    stale_threshold = (now - timedelta(minutes=STALE_THRESHOLD_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        async with await get_connection() as conn:
            async with conn.execute(
                """
                SELECT store_id, MAX(timestamp) as last_event_at
                FROM events
                GROUP BY store_id
                """
            ) as cur:
                store_rows = {row["store_id"]: row["last_event_at"] async for row in cur}

        for store_id in KNOWN_STORES:
            last_ts = store_rows.get(store_id)
            if not last_ts:
                feeds.append(StoreFeedStatus(
                    store_id=store_id,
                    last_event_at=None,
                    lag_seconds=None,
                    status="NO_DATA",
                ))
                continue

            # Compute lag
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                lag_s = (now - last_dt).total_seconds()
            except Exception:
                lag_s = None

            if last_ts < stale_threshold:
                status = "STALE_FEED"
            else:
                status = "OK"

            feeds.append(StoreFeedStatus(
                store_id=store_id,
                last_event_at=last_ts,
                lag_seconds=round(lag_s, 1) if lag_s is not None else None,
                status=status,
            ))

    except Exception as e:
        logger.error(f'"Health check query failed: {e}"')
        return JSONResponse(
            status_code=503,
            content={
                "status": "DEGRADED",
                "database": "UNAVAILABLE",
                "feeds": [],
                "checked_at": checked_at,
            },
        )

    overall_status = "DEGRADED" if any(f.status == "STALE_FEED" for f in feeds) else "OK"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        feeds=feeds,
        checked_at=checked_at,
    )
