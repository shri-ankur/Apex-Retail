# CHOICES.md — Key Architectural Decisions

## Decision 1: Detection Model — YOLOv8n

### Options Considered

| Model | Pros | Cons |
|---|---|---|
| YOLOv8n | Fast (CPU-viable), excellent person detection, simple API via `ultralytics` | Smaller than YOLOv8s/m — slightly lower accuracy on partially occluded persons |
| YOLOv8s | Better accuracy, still fast | ~2× slower than YOLOv8n; marginal improvement for retail densities |
| RT-DETR | Transformer-based, strong on crowded scenes | Requires GPU for real-time; complex dependency stack |
| MediaPipe | No GPU needed, runs on edge devices | Only detects one person per frame by default; useless for group entry |
| HOG+SVM (OpenCV) | Zero extra deps | Terrible on partial occlusion; no confidence scores |

### What AI Suggested

I asked Claude: *"For person detection in retail CCTV at 1080p/30fps, which YOLO variant gives the best accuracy/speed trade-off if I'm processing at 5fps on a CPU server?"*

Claude recommended YOLOv8s as a starting point, noting it achieves ~45 AP on COCO person class vs YOLOv8n's ~37 AP, and that 5fps processing gives enough headroom for the larger model. Claude also flagged RT-DETR as worth evaluating specifically for the group-entry edge case, because transformer attention handles overlapping bounding boxes better than anchor-based models.

### What I Chose and Why

**YOLOv8n** with the fallback to OpenCV HOG if ultralytics is unavailable.

My reasoning against Claude's YOLOv8s suggestion: the scoring harness runs in a containerised environment without GPU. YOLOv8n at 5fps on a 4-core CPU processes a 20-minute clip in ~8 minutes. YOLOv8s would take ~18 minutes. For a 48-hour challenge where I needed to iterate on tracker and Re-ID parameters, the faster cycle time was worth the accuracy trade-off.

For the partial occlusion edge case: I compensate at the tracker level rather than the detector level. Detections with confidence < 0.5 are not discarded — they are matched via centroid distance rather than IoU. A partially-occluded person with confidence 0.3 still gets tracked; it's just weighted less in the association step. This is better than relying on a larger model that might still miss the occlusion.

On the VLM question: I evaluated using Claude Vision for staff detection (classifying uniform colour per frame). I ran a test on 10 sample frames and found it correctly identified staff in 8/10 cases. I chose NOT to use it in the main pipeline because: (a) it adds latency of ~1.5s per frame, making real-time impossible; (b) the heuristic approach (presence duration + spatial variance) achieves similar results without API calls. The VLM would be valuable for one-time calibration — identifying what the staff uniform looks like in a new store — but not for per-frame inference.

---

## Decision 2: Event Schema Design

### The Core Tension

The challenge requires a single flat schema for all 8 event types, but different event types have different required fields. `BILLING_QUEUE_JOIN` needs `queue_depth`; `ZONE_DWELL` needs `dwell_ms`; `ENTRY` and `EXIT` need neither. The options were:

**Option A**: Separate schemas per event type (polymorphic). Clean types, strict validation, but complex ingest logic and schema evolution is harder.

**Option B**: Single flat schema with nullable fields (as specified). Universal ingest, easy deduplication, nullable fields for inapplicable events.

**Option C**: Single schema + strict validation rules (BILLING_QUEUE_JOIN must have queue_depth ≠ null). Best of both — catches data quality issues at ingest.

### What AI Suggested

I asked Claude: *"Should I enforce that BILLING_QUEUE_JOIN events have a non-null queue_depth at the schema validation level, or handle that downstream?"*

Claude suggested Option A (polymorphic schemas via Pydantic discriminated unions) for maximum type safety. The argument was that downstream analytics are more reliable when the schema enforces field presence.

### What I Chose and Why

**Option C** — single schema with field-level validation rules enforced in the `IngestResponse` error detail, not in the Pydantic model itself.

I disagreed with Claude's polymorphic approach because: the challenge spec explicitly provides a single schema. Diverging from it would fail the scoring harness's schema compliance check. More importantly, for a detection pipeline running on imperfect video, strict rejection of events with missing optional fields would create silent data loss. A `BILLING_QUEUE_JOIN` where the tracker couldn't determine queue depth should still be recorded with `queue_depth: null` — it's a detection confidence issue, not a schema error. The analytics layer handles nulls gracefully (excluded from averages, not counted in queue depth calculations).

The `metadata.session_seq` field was my addition to the spec — it's technically optional in the schema but I populate it on every event. It enables session reconstruction from a raw event stream without needing to join on `visitor_id` order by timestamp.

---

## Decision 3: API Architecture — Query-Time vs Pre-Aggregated Metrics

### The Problem

`GET /stores/{id}/metrics` must be "real-time — not cached from yesterday." There are two ways to achieve this:

**Option A: Query-time aggregation** — Every request queries raw events and aggregates on the fly. Guarantees real-time accuracy. Scales linearly with event count (slow at high volume).

**Option B: Pre-aggregated materialized views** — Ingest pipeline maintains running counters in a `metrics` table. Reads are O(1). But requires careful handling of corrections, re-ingests, and idempotency (updating a counter idempotently is non-trivial).

**Option C: Hybrid** — Aggregate in 1-minute rollup buckets on ingest. Read from rollups for historical data, query raw events only for the last 2 minutes. Best latency, moderate complexity.

### What AI Suggested

I described the scale (40 stores, ~500 events/hour per store) and asked Claude which approach to use. Claude recommended Option B (pre-aggregated) arguing that query-time aggregation would not scale. Claude suggested maintaining a `store_metrics_hourly` table updated on every ingest.

### What I Chose and Why

**Option A (query-time)** for this implementation, with a clear migration path to Option C documented in DESIGN.md.

I disagreed with Claude's pre-aggregation recommendation for this specific context:

1. **Scale**: 40 stores × 500 events/hour = 20,000 events/hour = ~333 events/minute. SQLite with WAL mode handles this trivially. Claude was applying advice appropriate for 10× this scale.

2. **Idempotency complexity**: The ingest endpoint is explicitly idempotent (duplicate event_ids are ignored). Pre-aggregated counters that are updated on ingest are very hard to make idempotent — you'd need to check if the event was already counted before incrementing. This is a correctness risk.

3. **Zero-traffic correctness**: The spec says "Handle zero-purchase stores. Real-time — not cached from yesterday." A pre-aggregated approach requires explicit handling of the "no events today" case. Query-time naturally returns zero for empty result sets.

4. **48-hour timeline**: Implementing and correctly testing a pre-aggregated system with idempotent counter updates would take 4+ hours. Query-time is implementable and testable in 1 hour.

The pre-aggregated approach would become necessary at ~10 stores × 5,000 events/hour. The DESIGN.md documents the migration path: add a `sessions` table populated on ingest, query that for funnel and metrics, keep raw events for anomaly detection only.
