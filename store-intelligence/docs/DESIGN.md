# DESIGN.md — Store Intelligence System Architecture

## Overview

The Store Intelligence system converts raw CCTV footage into a queryable real-time analytics API. It is built as two independent components:

1. **Detection Pipeline** (`pipeline/`) — Python scripts that process video frames, detect and track people, assign visitor session tokens, emit structured events to `.jsonl`
2. **Intelligence API** (`app/`) — FastAPI service that ingests those events, stores them, and exposes analytics endpoints

The two components communicate via the ingest endpoint: the pipeline can stream events directly to `POST /events/ingest` during live processing, or emit them to a `.jsonl` file for batch replay.

---

## Component Architecture

### Stage 1 — Detection Layer

```
Video frame → YOLOv8n detection → ByteTrack-style multi-object tracking → Zone mapping → Event emission
```

**Detection**: YOLOv8n is used for person detection. Frames are sampled at 5fps from a 30fps source — sufficient for retail movement speeds and reduces compute cost by 6×. Low-confidence detections (below threshold) are still emitted but flagged, not silently dropped. This ensures calibration data is preserved.

**Tracking**: A custom tracker (`tracker.py`) performs two-stage association:
- Stage 1: High-confidence detections matched to active tracks via IoU
- Stage 2: Remaining tracks + low-confidence detections matched via centroid distance and velocity prediction

**Re-ID**: Tracks that disappear are held in a `ReIDBuffer` for up to 10 seconds. New detections at the entry region are compared against the buffer by centroid proximity and bounding box size. Matching tracks get their `visitor_id` preserved and emit a `REENTRY` event rather than a new `ENTRY`. This directly addresses the re-entry inflation problem described in §3.3.

**Staff detection**: A heuristic classifier runs every 50 processed frames per track. Staff are characterised by: (a) presence across >40% of total video duration, (b) high spatial variance (moving across all store zones), (c) visiting ≥3 distinct zones. Staff tracks set `is_staff=true` on all their events and are excluded from all customer-facing metrics at query time.

**Zone mapping**: `ZoneMapper` converts pixel centroids to zone IDs using normalised bounding boxes per camera type. In production these regions would be calibrated against the store blueprint; for this implementation they use proportional defaults that can be overridden per-store in `store_layout.json`.

### Stage 2 — Event Stream

Events are emitted as newline-delimited JSON (`.jsonl`) conforming exactly to the schema specified in the challenge. Every event has a globally unique UUID v4 event_id. Timestamps are derived from the clip's known start time plus the frame offset (frame_index / native_fps), enabling accurate temporal correlation with POS data.

### Stage 3 — Intelligence API

FastAPI service with async SQLite backend (aiosqlite + WAL mode for concurrent read/write). All analytics are computed at query time on the raw events table — no pre-aggregated caches. This gives accurate real-time metrics but means queries scale with event volume. At 40 live stores, the `/funnel` endpoint would be the first to show latency degradation (see Scalability section).

**Idempotency**: `POST /events/ingest` uses `INSERT OR IGNORE` on the `event_id` primary key. Submitting the same batch twice has no effect and returns `duplicate` count in the response.

**Graceful degradation**: All endpoints catch DB exceptions and return structured HTTP 503 with `{"error": "DATABASE_UNAVAILABLE", "trace_id": "..."}`. No raw stack traces are ever exposed.

**Conversion correlation**: A visitor is counted as converted if they were in the `BILLING_COUNTER` zone within 5 minutes before a POS transaction timestamp for the same store. There is no customer_id in POS data, so time-window + store correlation is the only viable approach (per spec §3.4).

### Stage 4 — Live Dashboard

The `dashboard/live_dashboard.py` terminal dashboard uses the `rich` library to render a multi-panel layout that polls the API every 5 seconds: metrics, funnel, zone heatmap, active anomalies, and health status. This demonstrates that the pipeline and API are genuinely connected — the dashboard updates as events are ingested.

---

## AI-Assisted Decisions

### 1. Re-ID Strategy: Trajectory vs Appearance

I asked Claude (Sonnet): *"For Re-ID without torchreid, what's the minimum viable approach that handles the re-entry edge case correctly: same direction, 3-second gap, different person?"*

Claude suggested using colour histogram comparison of the upper body region as an appearance cue, combined with spatial proximity. I agreed with this direction but overrode the implementation detail: Claude's initial suggestion used a sliding window histogram over the full bounding box. I changed this to compare upper-body region only (top 40% of bbox), because lower body clothing (trousers, skirts) varies less between people than upper body in retail settings. The final `ReIDBuffer.find_match()` uses centroid distance + bbox size as a fast gate, with the colour histogram reserved for ambiguous cases. This runs faster and avoids the full-frame comparison Claude initially suggested.

### 2. Database Choice: SQLite vs PostgreSQL

I used Claude to reason through the trade-offs: SQLite is sufficient for a single-API-instance deployment handling 40 stores × ~500 events/hour = 20,000 events/hour. WAL mode in SQLite supports concurrent readers during writes, which is exactly the pattern we have (ingest writes + metrics reads). Claude suggested PostgreSQL for "production readiness." I disagreed: the challenge says `docker compose up` must be zero-config, and PostgreSQL adds a dependency that complicates this. I used SQLite with a note in CHOICES.md that Postgres migration requires only changing the `get_connection()` function and updating SQL syntax for `RETURNING` clauses.

### 3. Anomaly Detection Thresholds

I asked Claude: *"What are reasonable thresholds for billing queue spike detection in a retail context?"* Claude suggested queue depth > 3 as WARN and > 5 as CRITICAL, based on general retail literature. I adjusted to ≥5 as CRITICAL and kept the 2× 7-day average as a contextual check, because a queue of 5 may be normal for a busy Saturday but unusual on a Tuesday. The contextual check prevents false positives during expected peak periods.

---

## Scalability Considerations

At 40 live stores sending events in real time, the first endpoint to fail is `GET /stores/{id}/funnel`. The funnel query joins events against itself (via subqueries for visitor_id filtering) and against pos_transactions. As event volume grows, these self-joins become expensive. The fix is to materialise session-level aggregations in a separate `sessions` table on ingest, and query that instead of raw events.

The `/metrics` endpoint would be second to degrade because of the billing-POS correlation JOIN, which scans the full events table for the store. Partitioning by store_id (or switching to PostgreSQL table partitioning) would fix this.

For truly high-scale: replace the event store with a time-series database (InfluxDB, TimescaleDB) and pre-aggregate metrics into 1-minute rollup tables on ingest.
