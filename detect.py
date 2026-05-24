"""
detect.py — Main detection + tracking script for Store Intelligence pipeline.

Processes CCTV video clips using YOLOv8 for person detection and ByteTrack-style
multi-object tracking with custom Re-ID based on bounding box trajectory + appearance.

Design decisions:
  - YOLOv8n/s for detection (fast, accurate for person class)
  - Custom trajectory-based Re-ID (no torchreid dependency required)
  - Frame-skip processing at 5fps to balance accuracy vs speed
  - Zone mapping via configurable polygon regions
  - Staff detection via heuristic (position frequency + uniform color cues)
"""

import cv2
import json
import time
import uuid
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from typing import Optional

from tracker import MultiObjectTracker
from emit import EventEmitter
from zone_mapper import ZoneMapper

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"detect","msg":%(message)s}',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger(__name__)


def load_store_layout(layout_path: str, store_id: str) -> dict:
    with open(layout_path) as f:
        layout = json.load(f)
    for store in layout["stores"]:
        if store["store_id"] == store_id:
            return store
    raise ValueError(f"Store {store_id} not found in layout")


def get_clip_start_time(clip_timestamp: Optional[str], fallback_seconds: float = 0) -> datetime:
    """Parse clip start timestamp or construct from video creation metadata."""
    if clip_timestamp:
        return datetime.fromisoformat(clip_timestamp.replace("Z", "+00:00"))
    # Fallback: use current time minus video duration
    return datetime.now(timezone.utc)


def frame_to_timestamp(base_time: datetime, frame_idx: int, fps: float) -> str:
    """Convert frame index to ISO-8601 UTC timestamp."""
    offset_seconds = frame_idx / fps
    ts = base_time + timedelta(seconds=offset_seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def try_load_yolo():
    """Try to import YOLO; return model or None."""
    try:
        from ultralytics import YOLO
        logger.info('"Loading YOLOv8n model"')
        model = YOLO("yolov8n.pt")
        return model
    except ImportError:
        logger.warning('"ultralytics not installed; using OpenCV DNN fallback"')
        return None


def detect_persons_yolo(model, frame: np.ndarray, conf_threshold: float = 0.35):
    """Run YOLOv8 inference; return list of (bbox, confidence) for persons."""
    results = model(frame, classes=[0], conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3)
            })
    return detections


def detect_persons_dnn(net, frame: np.ndarray, conf_threshold: float = 0.35):
    """OpenCV DNN fallback for person detection (uses MobileNet SSD or HOG)."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    h, w = frame.shape[:2]
    scale = 640 / w
    small = cv2.resize(frame, (640, int(h * scale)))
    rects, weights = hog.detectMultiScale(
        small, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    detections = []
    for (x, y, bw, bh), w_val in zip(rects, weights):
        conf = float(min(w_val[0] / 2.0, 1.0))
        if conf >= conf_threshold:
            # Scale back
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3)
            })
    return detections


def estimate_staff_probability(track_history: dict, frame_w: int, frame_h: int) -> float:
    """
    Heuristic staff detection:
    - Staff appear in many frames (high dwell across whole video)
    - Staff appear near perimeter/back areas repeatedly
    - Staff movement is purposeful (cross multiple zones repeatedly)

    Returns probability 0.0 to 1.0 that this track is staff.
    """
    positions = track_history.get("positions", [])
    if len(positions) < 30:
        return 0.0

    # Staff heuristic 1: present in >40% of processed frames
    total_frames = track_history.get("total_frames_seen", 1)
    presence_ratio = len(positions) / max(total_frames, 1)
    staff_score = 0.0
    if presence_ratio > 0.4:
        staff_score += 0.5

    # Staff heuristic 2: high spatial variance (moves all over store)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    x_var = np.var(xs) / (frame_w ** 2)
    y_var = np.var(ys) / (frame_h ** 2)
    if (x_var + y_var) > 0.05:
        staff_score += 0.3

    # Staff heuristic 3: never dwells >2min in one zone (keeps moving)
    zone_visits = track_history.get("zone_visit_counts", {})
    if len(zone_visits) >= 3:
        staff_score += 0.2

    return min(staff_score, 1.0)


def process_video(
    video_path: str,
    store_layout: dict,
    camera_id: str,
    store_id: str,
    output_path: str,
    clip_start_time: Optional[str] = None,
    process_fps: float = 5.0,
    conf_threshold: float = 0.35,
    staff_threshold: float = 0.7
):
    """
    Main processing loop.

    Args:
        video_path: Path to .mp4 clip
        store_layout: Parsed store layout for this store
        camera_id: e.g. "CAM_ENTRY_01"
        store_id: e.g. "STORE_BLR_002"
        output_path: Where to write .jsonl events
        clip_start_time: ISO timestamp of clip start (used for event timestamps)
        process_fps: How many frames per second to process (default 5 — balance speed/accuracy)
        conf_threshold: Minimum detection confidence
        staff_threshold: Staff probability above which track is flagged as staff
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / native_fps

    logger.info(f'"Video: {video_path} | {frame_w}x{frame_h} @ {native_fps:.1f}fps | {duration_s:.0f}s | {total_frames} frames"')

    # Process every Nth frame to achieve target process_fps
    frame_step = max(1, int(native_fps / process_fps))
    base_time = get_clip_start_time(clip_start_time)

    # Load detection model
    yolo_model = try_load_yolo()
    dnn_net = None  # fallback

    # Initialize tracker, zone mapper, emitter
    tracker = MultiObjectTracker(
        max_disappeared=int(process_fps * 3),   # 3 seconds without detection before track dropped
        max_distance=150,                         # pixels: max IoU/centroid distance for association
        reid_window=int(process_fps * 10)         # Re-ID lookback window: 10 seconds
    )
    zone_mapper = ZoneMapper(store_layout, frame_w, frame_h, camera_id)
    emitter = EventEmitter(store_id, camera_id, output_path)

    frame_idx = 0
    processed_count = 0
    track_histories = defaultdict(lambda: {
        "positions": [],
        "zone_visit_counts": defaultdict(int),
        "total_frames_seen": 0,
        "first_seen_frame": None,
        "last_seen_frame": None,
    })

    # Zone dwell tracking: track_id -> {zone_id: start_frame}
    zone_dwell_start = {}     # track_id -> zone_id -> frame when entered
    zone_last_dwell_emitted = {}  # track_id -> zone_id -> last dwell emit frame

    logger.info(f'"Starting processing: step={frame_step}, process_fps={process_fps}"')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # --- Detection ---
        if yolo_model:
            detections = detect_persons_yolo(yolo_model, frame, conf_threshold)
        else:
            detections = detect_persons_dnn(dnn_net, frame, conf_threshold)

        # --- Tracking ---
        tracks = tracker.update(detections, frame_idx)

        # --- Per-track processing ---
        for track in tracks:
            track_id = track["track_id"]
            visitor_id = track["visitor_id"]
            bbox = track["bbox"]
            is_new = track.get("is_new", False)
            is_reentry = track.get("is_reentry", False)
            detection_conf = track.get("confidence", 0.5)

            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2

            # Update history
            h = track_histories[track_id]
            h["positions"].append((cx, cy))
            h["total_frames_seen"] += 1
            if h["first_seen_frame"] is None:
                h["first_seen_frame"] = frame_idx
            h["last_seen_frame"] = frame_idx

            # Staff classification (evaluated lazily — every 50 processed frames)
            is_staff = False
            if processed_count % 50 == 0:
                staff_prob = estimate_staff_probability(h, frame_w, frame_h)
                track["is_staff"] = staff_prob >= staff_threshold
            is_staff = track.get("is_staff", False)

            ts = frame_to_timestamp(base_time, frame_idx, native_fps)

            # Zone detection
            current_zone = zone_mapper.get_zone(cx, cy, camera_id)
            if current_zone:
                h["zone_visit_counts"][current_zone] += 1

            prev_zone = track.get("prev_zone")

            # --- Event emission ---
            session_seq = track.get("session_seq", 0)

            # ENTRY event
            if is_new:
                direction = zone_mapper.get_entry_direction(cy, frame_h)
                if direction == "INBOUND":
                    emitter.emit_entry(visitor_id, ts, is_staff, detection_conf, session_seq)
                    track["session_seq"] = session_seq + 1

            # REENTRY event
            if is_reentry:
                emitter.emit_reentry(visitor_id, ts, is_staff, detection_conf, session_seq)
                track["session_seq"] = session_seq + 1

            # Zone enter/exit events
            if current_zone != prev_zone:
                if prev_zone:
                    # Emit ZONE_EXIT for previous zone
                    dwell_ms = 0
                    if track_id in zone_dwell_start and prev_zone in zone_dwell_start[track_id]:
                        start_f = zone_dwell_start[track_id][prev_zone]
                        dwell_ms = int((frame_idx - start_f) / native_fps * 1000)
                        del zone_dwell_start[track_id][prev_zone]

                    emitter.emit_zone_exit(visitor_id, ts, prev_zone, dwell_ms, is_staff, detection_conf, session_seq)
                    track["session_seq"] = track.get("session_seq", 0) + 1

                    # Billing queue abandon detection
                    if prev_zone == "BILLING_COUNTER":
                        # Check if a POS transaction followed (simplified: emit abandon if leaving billing without purchase signal)
                        emitter.emit_billing_queue_abandon(visitor_id, ts, is_staff, detection_conf, session_seq)
                        track["session_seq"] = track.get("session_seq", 0) + 1

                if current_zone:
                    # Emit ZONE_ENTER for new zone
                    queue_depth = tracker.get_zone_occupancy("BILLING_COUNTER") if current_zone == "BILLING_COUNTER" else None
                    emitter.emit_zone_enter(visitor_id, ts, current_zone, is_staff, detection_conf, session_seq, queue_depth)
                    track["session_seq"] = track.get("session_seq", 0) + 1

                    # Track dwell start
                    if track_id not in zone_dwell_start:
                        zone_dwell_start[track_id] = {}
                    zone_dwell_start[track_id][current_zone] = frame_idx

                    # Billing queue join
                    if current_zone == "BILLING_COUNTER" and queue_depth and queue_depth > 0:
                        emitter.emit_billing_queue_join(visitor_id, ts, queue_depth, is_staff, detection_conf, session_seq)
                        track["session_seq"] = track.get("session_seq", 0) + 1

            # ZONE_DWELL — every 30s of continued dwell
            if current_zone and track_id in zone_dwell_start and current_zone in zone_dwell_start.get(track_id, {}):
                start_f = zone_dwell_start[track_id][current_zone]
                dwell_s = (frame_idx - start_f) / native_fps
                last_emit_f = zone_last_dwell_emitted.get(track_id, {}).get(current_zone, start_f)
                seconds_since_last_emit = (frame_idx - last_emit_f) / native_fps

                if dwell_s >= 30 and seconds_since_last_emit >= 30:
                    emitter.emit_zone_dwell(visitor_id, ts, current_zone, int(dwell_s * 1000), is_staff, detection_conf, session_seq)
                    track["session_seq"] = track.get("session_seq", 0) + 1
                    if track_id not in zone_last_dwell_emitted:
                        zone_last_dwell_emitted[track_id] = {}
                    zone_last_dwell_emitted[track_id][current_zone] = frame_idx

            track["prev_zone"] = current_zone

        # --- Handle disappeared tracks (EXIT events) ---
        for disappeared_id, track_info in tracker.get_just_disappeared():
            visitor_id = track_info["visitor_id"]
            ts = frame_to_timestamp(base_time, frame_idx, native_fps)
            is_staff = track_info.get("is_staff", False)
            conf = track_info.get("confidence", 0.5)
            seq = track_info.get("session_seq", 0)
            emitter.emit_exit(visitor_id, ts, is_staff, conf, seq)

        processed_count += 1
        frame_idx += 1

        if processed_count % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            logger.info(f'"Progress: {progress:.1f}% | Processed frames: {processed_count} | Active tracks: {len(tracker.active_tracks)}"')

    cap.release()
    emitter.flush()

    stats = emitter.get_stats()
    logger.info(f'"Processing complete: {json.dumps(stats)}"')
    return stats


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to .mp4 CCTV clip")
    parser.add_argument("--store-id", default="STORE_BLR_002", help="Store ID from store_layout.json")
    parser.add_argument("--camera-id", default="CAM_ENTRY_01", help="Camera ID")
    parser.add_argument("--layout", default="store_layout.json", help="Path to store_layout.json")
    parser.add_argument("--output", default="events_output.jsonl", help="Output .jsonl file for events")
    parser.add_argument("--clip-start", default=None, help="ISO-8601 clip start time (e.g. 2026-03-03T14:00:00Z)")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to process (default: 5)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    args = parser.parse_args()

    store_layout = load_store_layout(args.layout, args.store_id)

    stats = process_video(
        video_path=args.video,
        store_layout=store_layout,
        camera_id=args.camera_id,
        store_id=args.store_id,
        output_path=args.output,
        clip_start_time=args.clip_start,
        process_fps=args.fps,
        conf_threshold=args.conf,
    )

    print(f"\n✅ Detection complete. Events written to: {args.output}")
    print(f"   Total events emitted: {stats['total_events']}")
    print(f"   Unique visitors: {stats['unique_visitors']}")
    print(f"   Staff tracks filtered: {stats['staff_tracks']}")


if __name__ == "__main__":
    main()
