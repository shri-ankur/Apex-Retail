#!/usr/bin/env bash
# run.sh — Process all 5 CCTV clips and optionally ingest into the API.
#
# Camera mapping (from frame inspection):
#   CAM_1.mp4  → CAM_FLOOR_A      Main floor, skincare/cleanser wall
#   CAM_2.mp4  → CAM_FLOOR_B      Main floor, makeup/colour cosmetics wall
#   CAM_3.mp4  → CAM_ENTRY_01     Entry/exit threshold (glass door, top-down)
#   CAM_4.mp4  → CAM_STOCKROOM_01 Back stockroom (staff-only, HEVC 25fps)
#   CAM_5.mp4  → CAM_BILLING_01   Billing/POS counter (HEVC 25fps)
#
# Usage:
#   ./pipeline/run.sh                              # process only
#   ./pipeline/run.sh http://localhost:8000        # process + ingest into API
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/.."
LAYOUT="${ROOT}/store_layout.json"
OUTPUT_DIR="${ROOT}/events"
VIDEO_DIR="${ROOT}/videos"
API_ENDPOINT="${1:-}"

mkdir -p "$OUTPUT_DIR"

echo "🔍 Store Intelligence — Detection Pipeline"
echo "==========================================="
echo "Store:     STORE_BLR_002 (Purplle Beauty, Bangalore)"
echo "Videos:    $VIDEO_DIR"
echo "Output:    $OUTPUT_DIR"
echo "API:       ${API_ENDPOINT:-'(not configured — events written to disk only)'}"
echo ""

# Camera config: FILE|CAMERA_ID|FPS|CLIP_START_UTC|NOTES
CAMERAS=(
  "CAM_3.mp4|CAM_ENTRY_01|29.97|2026-04-10T14:40:32Z|Entry/exit threshold"
  "CAM_1.mp4|CAM_FLOOR_A|29.97|2026-04-10T14:40:57Z|Main floor - skincare wall"
  "CAM_2.mp4|CAM_FLOOR_B|29.97|2026-04-10T14:40:32Z|Main floor - makeup wall"
  "CAM_5.mp4|CAM_BILLING_01|25.0|2026-04-10T14:40:18Z|Billing/POS counter"
  "CAM_4.mp4|CAM_STOCKROOM_01|25.0|2026-04-10T14:40:15Z|Stockroom (staff-only)"
)

TOTAL_EVENTS=0

for ENTRY in "${CAMERAS[@]}"; do
  IFS='|' read -r VIDEO_FILE CAMERA_ID FPS CLIP_START NOTES <<< "$ENTRY"
  VIDEO_PATH="$VIDEO_DIR/$VIDEO_FILE"
  OUTPUT_FILE="$OUTPUT_DIR/${CAMERA_ID}_events.jsonl"

  if [ ! -f "$VIDEO_PATH" ]; then
    echo "⚠️  Skipping $VIDEO_FILE (not found at $VIDEO_PATH)"
    continue
  fi

  echo "📹 $VIDEO_FILE → $CAMERA_ID  [$NOTES]"
  python3 "$SCRIPT_DIR/detect.py" \
    --video "$VIDEO_PATH" \
    --store-id "STORE_BLR_002" \
    --camera-id "$CAMERA_ID" \
    --layout "$LAYOUT" \
    --output "$OUTPUT_FILE" \
    --clip-start "$CLIP_START" \
    --fps 5.0 \
    --conf 0.35

  COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
  TOTAL_EVENTS=$((TOTAL_EVENTS + COUNT))
  echo "   ✅ $COUNT events → $OUTPUT_FILE"
  echo ""
done

# Merge all event files into one
MERGED="$OUTPUT_DIR/all_events.jsonl"
cat "$OUTPUT_DIR"/*_events.jsonl 2>/dev/null | sort > "$MERGED" || true
echo "📦 Merged: $MERGED  ($TOTAL_EVENTS total events)"

# Ingest into API if endpoint provided
if [ -n "$API_ENDPOINT" ]; then
  echo ""
  echo "⚡ Ingesting into API at $API_ENDPOINT..."
  python3 - <<PYTHON
import json, requests, sys

BATCH_SIZE = 500
events = []
with open("$MERGED") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

total = len(events)
ingested = 0
errors = 0

for i in range(0, total, BATCH_SIZE):
    batch = events[i:i+BATCH_SIZE]
    try:
        resp = requests.post(
            "$API_ENDPOINT/events/ingest",
            json={"events": batch},
            timeout=10
        )
        if resp.status_code == 200:
            body = resp.json()
            ingested += body.get("accepted", 0)
            print(f"  Batch {i//BATCH_SIZE + 1}: accepted={body.get('accepted')} duplicate={body.get('duplicate')} rejected={body.get('rejected')}")
        else:
            errors += len(batch)
            print(f"  ERROR batch {i//BATCH_SIZE + 1}: HTTP {resp.status_code}", file=sys.stderr)
    except Exception as e:
        errors += len(batch)
        print(f"  ERROR batch {i//BATCH_SIZE + 1}: {e}", file=sys.stderr)

print(f"\n✅ Ingested {ingested}/{total} events  ({errors} errors)")
PYTHON
fi

echo ""
echo "🎉 Pipeline complete!"
