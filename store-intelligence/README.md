# Store Intelligence — CCTV to Live Retail Analytics

End-to-end pipeline: raw CCTV footage → real-time store analytics API.

## Quick Start (5 commands)

```bash
# 1. Clone and enter repo
git clone <your-repo-url> store-intelligence && cd store-intelligence

# 2. Copy all 5 CCTV clips into the videos/ directory
mkdir -p videos events
cp /path/to/CAM_1.mp4 videos/   # Main floor A — skincare wall
cp /path/to/CAM_2.mp4 videos/   # Main floor B — makeup wall
cp /path/to/CAM_3.mp4 videos/   # Entry/exit threshold
cp /path/to/CAM_4.mp4 videos/   # Stockroom (staff-only, HEVC)
cp /path/to/CAM_5.mp4 videos/   # Billing/POS counter (HEVC)

# 3. Start the API (builds container, initialises DB)
docker compose up -d

# 4. Run the detection pipeline against the clips
pip install -r requirements.txt
bash pipeline/run.sh --api-endpoint http://localhost:8000

# 5. Launch the live dashboard
python dashboard/live_dashboard.py --api http://localhost:8000 --store STORE_BLR_002
```

The API is now live at **http://localhost:8000** — docs at **http://localhost:8000/docs**.

---

## Camera Inventory

All 5 cameras cover a single **Purplle Beauty** store in Bangalore. Mapped from frame inspection:

| File | Camera ID | Type | Codec | FPS | What It Shows |
|---|---|---|---|---|---|
| CAM_1.mp4 | CAM_FLOOR_A | main_floor | H264 | 29.97 | Skincare/cleanser wall (The Face Shop, Minimalist, Dermaco, COSRX) |
| CAM_2.mp4 | CAM_FLOOR_B | main_floor | H264 | 29.97 | Makeup wall (L'Oreal, Swiss Beauty, Lakme, Maybelline) + seasonal display |
| CAM_3.mp4 | CAM_ENTRY_01 | entry_exit | H264 | 29.97 | Glass door entry threshold — ENTRY/EXIT events sourced here |
| CAM_4.mp4 | CAM_STOCKROOM_01 | stockroom | **HEVC** | 25 | Back stockroom — staff-only, all detections is_staff=True |
| CAM_5.mp4 | CAM_BILLING_01 | billing | **HEVC** | 25 | Billing/POS counter — laptop terminal, queue detection |

> **HEVC note:** CAM_4 and CAM_5 are H.265 encoded. The pipeline uses an ffmpeg pipe fallback if OpenCV cannot hardware-decode HEVC on the host system.

## Running Detection Against a Single Clip

```bash
# Entry camera (H264)
python pipeline/detect.py \
  --video videos/CAM_3.mp4 \
  --store-id STORE_BLR_002 \
  --camera-id CAM_ENTRY_01 \
  --layout store_layout.json \
  --output events/CAM_ENTRY_01_events.jsonl \
  --clip-start 2026-04-10T14:40:32Z \
  --fps 5.0 --conf 0.35

# Billing camera (HEVC — ffmpeg fallback used automatically)
python pipeline/detect.py \
  --video videos/CAM_5.mp4 \
  --store-id STORE_BLR_002 \
  --camera-id CAM_BILLING_01 \
  --layout store_layout.json \
  --output events/CAM_BILLING_01_events.jsonl \
  --clip-start 2026-04-10T14:40:18Z \
  --fps 5.0 --conf 0.35
```

Events are written as newline-delimited JSON (one event per line).

## Ingesting Events into the API

```bash
# Batch ingest from jsonl file
python - <<'EOF'
import json, requests

with open("events/CAM_ENTRY_01_events.jsonl") as f:
    events = [json.loads(l) for l in f if l.strip()]

# Send in batches of 500
for i in range(0, len(events), 500):
    batch = events[i:i+500]
    r = requests.post("http://localhost:8000/events/ingest", json={"events": batch})
    print(f"Batch {i//500+1}: {r.json()}")
EOF
```

Or use the all-in-one script which processes all clips AND ingests:

```bash
bash pipeline/run.sh --api-endpoint http://localhost:8000
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /events/ingest` | Ingest up to 500 events per batch |
| `GET /stores/{id}/metrics` | Real-time visitor, conversion, dwell metrics |
| `GET /stores/{id}/funnel` | Entry → Zone → Billing → Purchase funnel |
| `GET /stores/{id}/heatmap` | Zone visit frequency + dwell heatmap |
| `GET /stores/{id}/anomalies` | Active anomaly detection |
| `GET /health` | Service health + feed staleness check |

### Example Requests

```bash
# Ingest events
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id":"...", "store_id":"STORE_BLR_002", ...}]}'

# Get metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics | jq .

# Get funnel
curl http://localhost:8000/stores/STORE_BLR_002/funnel | jq .

# Check health
curl http://localhost:8000/health | jq .
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v --tb=short
```

Expected coverage: >70% statement coverage across `app/`.

---

## Architecture

```
CCTV Clips (.mp4)
      │
      ▼
pipeline/detect.py   ← YOLOv8n detection + ByteTrack-style tracking
      │                 Re-ID via trajectory + appearance
      ▼
events/*.jsonl       ← Structured events (schema: event_id, visitor_id, event_type, ...)
      │
      ▼
POST /events/ingest  ← Idempotent batch ingest (SQLite, WAL mode)
      │
      ├─► GET /stores/{id}/metrics    ← Unique visitors, conversion rate, dwell, queue
      ├─► GET /stores/{id}/funnel     ← Entry → Zone → Billing → Purchase
      ├─► GET /stores/{id}/heatmap    ← Zone frequency + dwell, normalised 0–100
      ├─► GET /stores/{id}/anomalies  ← Queue spike, conversion drop, dead zone
      └─► GET /health                 ← DB + feed staleness
              │
              ▼
      dashboard/live_dashboard.py  ← Terminal UI, refreshes every 5s
```

See [docs/DESIGN.md](docs/DESIGN.md) for full architecture and AI-assisted decisions.
See [docs/CHOICES.md](docs/CHOICES.md) for model selection, schema design, and API architecture rationale.

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py       # Main detection + tracking script
│   ├── tracker.py      # Re-ID / multi-object tracking
│   ├── emit.py         # Event schema + emission
│   ├── zone_mapper.py  # Pixel → zone mapping
│   └── run.sh          # One command: all clips → events → API
├── app/
│   ├── main.py         # FastAPI entrypoint + middleware
│   ├── models.py       # Pydantic event schema + response models
│   ├── database.py     # Async SQLite (aiosqlite)
│   ├── ingestion.py    # POST /events/ingest
│   ├── metrics.py      # GET /stores/{id}/metrics
│   ├── funnel.py       # GET /stores/{id}/funnel
│   ├── heatmap.py      # GET /stores/{id}/heatmap
│   ├── anomalies.py    # GET /stores/{id}/anomalies
│   └── health.py       # GET /health
├── dashboard/
│   └── live_dashboard.py  # Rich terminal dashboard (Part E)
├── tests/
│   └── test_api.py     # Full test suite with edge cases
├── docs/
│   ├── DESIGN.md       # Architecture + AI-assisted decisions
│   └── CHOICES.md      # 3 decisions with full reasoning
├── store_layout.json
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.dashboard
├── requirements.txt
└── README.md
```

---

## Live Dashboard (Part E)

```bash
# Terminal dashboard — runs against the live API
python dashboard/live_dashboard.py --api http://localhost:8000 --store STORE_BLR_002
```

The dashboard auto-refreshes every 5 seconds and shows:
- Live visitor count, conversion rate, revenue, queue depth
- Conversion funnel with drop-off percentages
- Zone heatmap with heat bars
- Active anomalies with severity and suggested actions
- API health + feed staleness

Local URL: **terminal** (no browser required; runs in any terminal ≥ 80 columns wide)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `/data/store_intelligence.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_URL` | `http://localhost:8000` | API URL for dashboard |
| `STORE_ID` | `STORE_BLR_002` | Default store for dashboard |
