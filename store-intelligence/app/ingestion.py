"""
ingestion.py — Event ingest endpoint with idempotency, validation, deduplication.

POST /events/ingest
  - Accepts up to 500 events per batch
  - Idempotent: calling twice with same payload produces same result (event_id dedup)
  - Partial success: malformed events are rejected with per-event error detail
  - Structured error responses (no raw stack traces)
"""

import json
import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import IngestRequest, IngestResponse, StoreEvent
from database import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


async def _insert_events(conn, valid_events: List[StoreEvent]) -> tuple[int, int]:
    """
    Bulk insert events with conflict ignore (idempotency via PRIMARY KEY event_id).
    Returns (inserted_count, duplicate_count).
    """
    rows = [
        (
            e.event_id,
            e.store_id,
            e.camera_id,
            e.visitor_id,
            e.event_type,
            e.timestamp,
            e.zone_id,
            e.dwell_ms,
            1 if e.is_staff else 0,
            e.confidence,
            e.metadata.queue_depth,
            e.metadata.sku_zone,
            e.metadata.session_seq,
        )
        for e in valid_events
    ]

    inserted = 0
    duplicate = 0

    for row in rows:
        try:
            await conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_id, store_id, camera_id, visitor_id, event_type,
                    timestamp, zone_id, dwell_ms, is_staff, confidence,
                    queue_depth, sku_zone, session_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            if conn.total_changes > 0:
                inserted += 1
            else:
                duplicate += 1
        except Exception as e:
            logger.warning(f'"Failed to insert event {row[0]}: {e}"')

    await conn.commit()
    return inserted, duplicate


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(request: Request):
    """
    Ingest a batch of store events (up to 500 per call).

    Idempotent: re-submitting the same event_ids has no effect.
    Returns:
      - accepted: number of events written to DB
      - duplicate: number of already-seen event_ids skipped
      - rejected: number of events that failed schema validation
      - errors: per-event error detail for rejected events
    """
    trace_id = getattr(request.state, "trace_id", "unknown")

    # Parse raw body for partial-success handling
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_JSON", "message": "Request body must be valid JSON"},
        )

    raw_events = body.get("events", [])
    if not isinstance(raw_events, list):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_FORMAT", "message": "'events' must be an array"},
        )

    if len(raw_events) > 500:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "BATCH_TOO_LARGE",
                "message": f"Maximum batch size is 500 events. Got {len(raw_events)}.",
            },
        )

    # Validate each event individually for partial success
    valid_events: List[StoreEvent] = []
    errors = []

    for i, raw in enumerate(raw_events):
        try:
            event = StoreEvent.model_validate(raw)
            valid_events.append(event)
        except ValidationError as e:
            event_id = raw.get("event_id", f"<index:{i}>")
            errors.append({
                "index": i,
                "event_id": event_id,
                "errors": e.errors(include_url=False),
            })

    accepted = 0
    duplicate = 0

    if valid_events:
        try:
            async with await get_connection() as conn:
                inserted, dup = await _insert_events(conn, valid_events)
                accepted = inserted
                duplicate = dup
        except Exception as e:
            logger.error(f'"DB write failed trace_id={trace_id}: {e}"')
            return JSONResponse(
                status_code=503,
                content={
                    "error": "DATABASE_UNAVAILABLE",
                    "message": "Event store is temporarily unavailable. Please retry.",
                    "trace_id": trace_id,
                },
            )

    logger.info(
        f'{{"trace_id":"{trace_id}","endpoint":"ingest","total":{len(raw_events)},'
        f'"accepted":{accepted},"duplicate":{duplicate},"rejected":{len(errors)}}}'
    )

    return IngestResponse(
        accepted=accepted,
        rejected=len(errors),
        duplicate=duplicate,
        errors=errors,
    )
