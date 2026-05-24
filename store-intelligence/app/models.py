"""
models.py — Pydantic event schema and API response models.

The event schema mirrors the required output schema exactly.
All fields are validated on ingest; malformed events return structured errors.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from uuid import UUID
import uuid


VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
    "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
}

VALID_STORE_IDS = {
    "STORE_BLR_002", "STORE_BLR_001", "STORE_MUM_001",
    "STORE_DEL_001", "STORE_HYD_001"
}

# Camera IDs derived from actual footage analysis
VALID_CAMERA_IDS = {
    "CAM_ENTRY_01",      # CAM_3.mp4  — entry/exit threshold (H264, 29.97fps)
    "CAM_FLOOR_A",       # CAM_1.mp4  — skincare/cleanser wall (H264, 29.97fps)
    "CAM_FLOOR_B",       # CAM_2.mp4  — makeup/colour cosmetics wall (H264, 29.97fps)
    "CAM_BILLING_01",    # CAM_5.mp4  — billing/POS counter (HEVC, 25fps)
    "CAM_STOCKROOM_01",  # CAM_4.mp4  — back stockroom, staff-only (HEVC, 25fps)
}

# Zone IDs from store_layout.json (updated for actual store)
VALID_ZONE_IDS = {
    "ENTRY_THRESHOLD",
    "SKINCARE",
    "MAKEUP",
    "SEASONAL_DISPLAY",
    "ACCESSORIES",
    "BEAUTY_ADVISOR_DESK",
    "BILLING_COUNTER",
    "STOCKROOM",
}


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0

    model_config = {"extra": "allow"}


class StoreEvent(BaseModel):
    event_id: str = Field(..., description="UUID v4 — must be globally unique")
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{v}'. Must be one of: {sorted(VALID_EVENT_TYPES)}")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        try:
            # Accept Z or +00:00 suffix
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid timestamp format '{v}'. Expected ISO-8601 UTC (e.g. 2026-03-03T14:22:10Z)")
        return v

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v):
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"event_id must be a valid UUID v4, got: '{v}'")
        return v

    @field_validator("dwell_ms")
    @classmethod
    def validate_dwell_ms(cls, v):
        if v < 0:
            raise ValueError("dwell_ms cannot be negative")
        return v

    model_config = {"extra": "forbid"}


class IngestRequest(BaseModel):
    events: List[StoreEvent] = Field(..., max_length=500, description="Up to 500 events per batch")


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    duplicate: int
    errors: List[dict] = []


# --- Metrics Response Models ---

class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_seconds: float
    visitor_count: int


class StoreMetricsResponse(BaseModel):
    store_id: str
    date: str
    unique_visitors: int
    conversion_rate: float = Field(..., description="Ratio: visitors who purchased / total unique visitors")
    avg_dwell_per_zone: List[ZoneDwellMetric]
    current_queue_depth: int
    abandonment_rate: float = Field(..., description="Ratio: billing abandonments / billing queue joins")
    total_revenue_inr: float
    as_of: str  # Timestamp of last event processed


# --- Funnel Response Models ---

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    funnel: List[FunnelStage]
    session_window: str


# --- Heatmap Response Models ---

class ZoneHeatmapEntry(BaseModel):
    zone_id: str
    visit_frequency: int
    avg_dwell_seconds: float
    normalised_score: float = Field(..., ge=0.0, le=100.0)
    data_confidence: str = Field(..., description="HIGH | LOW (LOW if <20 sessions in window)")


class HeatmapResponse(BaseModel):
    store_id: str
    zones: List[ZoneHeatmapEntry]
    generated_at: str


# --- Anomaly Response Models ---

class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: str
    severity: str  # INFO | WARN | CRITICAL
    description: str
    suggested_action: str
    detected_at: str
    metadata: dict = {}


class AnomalyResponse(BaseModel):
    store_id: str
    active_anomalies: List[Anomaly]
    checked_at: str


# --- Health Response ---

class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_at: Optional[str]
    lag_seconds: Optional[float]
    status: str  # OK | STALE_FEED | NO_DATA


class HealthResponse(BaseModel):
    status: str  # OK | DEGRADED
    database: str  # OK | UNAVAILABLE
    feeds: List[StoreFeedStatus]
    checked_at: str
