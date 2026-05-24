"""
emit.py — Event schema definition and structured event emission.

All events conform to the schema defined in the challenge instructions.
Events are written to a .jsonl file (one JSON object per line) and optionally
posted to the ingest API endpoint for live processing.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def make_event_id() -> str:
    return str(uuid.uuid4())


class EventEmitter:
    """
    Builds and emits structured events to .jsonl output.

    All events match the schema:
    {
        "event_id": "uuid-v4",
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_c8a2f1",
        "event_type": "ZONE_DWELL",
        "timestamp": "2026-03-03T14:22:10Z",
        "zone_id": "SKINCARE",
        "dwell_ms": 8400,
        "is_staff": false,
        "confidence": 0.91,
        "metadata": {
            "queue_depth": null,
            "sku_zone": "MOISTURISER",
            "session_seq": 5
        }
    }
    """

    VALID_EVENT_TYPES = {
        "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
        "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
    }

    # SKU zone labels mapped from real zone IDs observed in footage
    # (store sells: The Face Shop, Minimalist, Dermaco, L'Oreal, Lakme, Maybelline, Swiss Beauty)
    ZONE_SKU_MAP = {
        "SKINCARE":            "SKINCARE",
        "MAKEUP":              "MAKEUP",
        "SEASONAL_DISPLAY":    "PROMO",
        "ACCESSORIES":         "ACCESSORIES",
        "BEAUTY_ADVISOR_DESK": None,
        "BILLING_COUNTER":     None,
        "ENTRY_THRESHOLD":     None,
        "STOCKROOM":           None,
    }

    def __init__(self, store_id: str, camera_id: str, output_path: str, api_endpoint: Optional[str] = None):
        self.store_id = store_id
        self.camera_id = camera_id
        self.output_path = output_path
        self.api_endpoint = api_endpoint
        self._buffer = []
        self._stats = {
            "total_events": 0,
            "unique_visitors": set(),
            "staff_tracks": 0,
            "events_by_type": {}
        }
        self._file = open(output_path, "w", encoding="utf-8")

    def _build_event(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: str,
        zone_id: Optional[str],
        dwell_ms: int,
        is_staff: bool,
        confidence: float,
        session_seq: int,
        queue_depth: Optional[int] = None,
    ) -> dict:
        assert event_type in self.VALID_EVENT_TYPES, f"Invalid event type: {event_type}"

        sku_zone = self.ZONE_SKU_MAP.get(zone_id) if zone_id else None

        event = {
            "event_id": make_event_id(),
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(float(confidence), 3),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": sku_zone,
                "session_seq": session_seq,
            }
        }
        return event

    def _emit(self, event: dict):
        """Write event to jsonl and update stats."""
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()

        self._stats["total_events"] += 1
        self._stats["unique_visitors"].add(event["visitor_id"])
        etype = event["event_type"]
        self._stats["events_by_type"][etype] = self._stats["events_by_type"].get(etype, 0) + 1
        if event.get("is_staff"):
            if etype == "ENTRY":
                self._stats["staff_tracks"] += 1

        # Optional: POST to live API
        if self.api_endpoint:
            try:
                import requests
                requests.post(
                    f"{self.api_endpoint}/events/ingest",
                    json={"events": [event]},
                    timeout=2
                )
            except Exception as e:
                logger.warning(f'"Failed to post event to API: {e}"')

    def emit_entry(self, visitor_id: str, timestamp: str, is_staff: bool, confidence: float, session_seq: int):
        event = self._build_event(
            visitor_id=visitor_id, event_type="ENTRY", timestamp=timestamp,
            zone_id=None, dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def emit_exit(self, visitor_id: str, timestamp: str, is_staff: bool, confidence: float, session_seq: int):
        event = self._build_event(
            visitor_id=visitor_id, event_type="EXIT", timestamp=timestamp,
            zone_id=None, dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def emit_reentry(self, visitor_id: str, timestamp: str, is_staff: bool, confidence: float, session_seq: int):
        event = self._build_event(
            visitor_id=visitor_id, event_type="REENTRY", timestamp=timestamp,
            zone_id=None, dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def emit_zone_enter(
        self, visitor_id: str, timestamp: str, zone_id: str,
        is_staff: bool, confidence: float, session_seq: int,
        queue_depth: Optional[int] = None
    ):
        event = self._build_event(
            visitor_id=visitor_id, event_type="ZONE_ENTER", timestamp=timestamp,
            zone_id=zone_id, dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq, queue_depth=queue_depth
        )
        self._emit(event)

    def emit_zone_exit(
        self, visitor_id: str, timestamp: str, zone_id: str,
        dwell_ms: int, is_staff: bool, confidence: float, session_seq: int
    ):
        event = self._build_event(
            visitor_id=visitor_id, event_type="ZONE_EXIT", timestamp=timestamp,
            zone_id=zone_id, dwell_ms=dwell_ms, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def emit_zone_dwell(
        self, visitor_id: str, timestamp: str, zone_id: str,
        dwell_ms: int, is_staff: bool, confidence: float, session_seq: int
    ):
        event = self._build_event(
            visitor_id=visitor_id, event_type="ZONE_DWELL", timestamp=timestamp,
            zone_id=zone_id, dwell_ms=dwell_ms, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def emit_billing_queue_join(
        self, visitor_id: str, timestamp: str, queue_depth: int,
        is_staff: bool, confidence: float, session_seq: int
    ):
        event = self._build_event(
            visitor_id=visitor_id, event_type="BILLING_QUEUE_JOIN", timestamp=timestamp,
            zone_id="BILLING_COUNTER", dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq, queue_depth=queue_depth
        )
        self._emit(event)

    def emit_billing_queue_abandon(
        self, visitor_id: str, timestamp: str,
        is_staff: bool, confidence: float, session_seq: int
    ):
        event = self._build_event(
            visitor_id=visitor_id, event_type="BILLING_QUEUE_ABANDON", timestamp=timestamp,
            zone_id="BILLING_COUNTER", dwell_ms=0, is_staff=is_staff, confidence=confidence,
            session_seq=session_seq
        )
        self._emit(event)

    def flush(self):
        self._file.flush()
        self._file.close()

    def get_stats(self) -> dict:
        return {
            "total_events": self._stats["total_events"],
            "unique_visitors": len(self._stats["unique_visitors"]),
            "staff_tracks": self._stats["staff_tracks"],
            "events_by_type": self._stats["events_by_type"],
        }
