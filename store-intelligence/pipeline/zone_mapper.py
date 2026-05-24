"""
zone_mapper.py — Maps pixel coordinates to named store zones.

Zone regions are calibrated from actual frame inspection of each camera:

CAM_ENTRY_01 (CAM_3.mp4) — Top-down entry threshold
  - The glass door is in the center-bottom of the frame
  - Interior floor (wood) is top-half; exterior (black tile) is bottom-half
  - Entry direction: centroid moving from bottom → top = INBOUND
  - Exit direction: centroid moving from top → bottom = OUTBOUND

CAM_FLOOR_A (CAM_1.mp4) — Main floor, skincare wall
  - Full frame is sales floor; subdivided into product zones vs consultation desk
  - Left strip (~0–25% width): beauty advisor desk / mirror station
  - Main area (25–100% width): skincare shelves

CAM_FLOOR_B (CAM_2.mp4) — Main floor, makeup wall
  - Left edge (0–15%): accessories section
  - Center-bottom: seasonal/promotional display island
  - Right wall: makeup brands (full height)

CAM_BILLING_01 (CAM_5.mp4) — POS counter
  - Staff at counter occupies left-center
  - Customer approach zone: right-of-center approaching counter
  - Entire frame treated as BILLING_COUNTER zone

CAM_STOCKROOM_01 (CAM_4.mp4) — Staff-only back room
  - Entire frame = STOCKROOM; all detections → is_staff=True
"""

from typing import Optional


# Normalised zone rectangles per camera: {zone_id: (x1_r, y1_r, x2_r, y2_r)}
# Ratios relative to (frame_w, frame_h). Calibrated from actual frame analysis.

CAMERA_ZONE_MAP = {
    # ── CAM_ENTRY_01 (CAM_3.mp4) ─────────────────────────────────────────────
    # Top-down view of glass door + foyer
    # Interior wood floor visible in top 60%; exterior black tile in bottom 40%
    "CAM_ENTRY_01": {
        "ENTRY_THRESHOLD": (0.1, 0.3, 0.9, 0.85),  # The doorway crossing zone
    },

    # ── CAM_FLOOR_A (CAM_1.mp4) ──────────────────────────────────────────────
    # Skincare/cleanser wall + beauty advisor desk
    "CAM_FLOOR_A": {
        "BEAUTY_ADVISOR_DESK": (0.0,  0.3,  0.25, 0.85),   # Left strip — mirror station
        "SKINCARE":            (0.25, 0.0,  1.0,  0.75),   # Main shelving area
    },

    # ── CAM_FLOOR_B (CAM_2.mp4) ──────────────────────────────────────────────
    # Makeup wall + seasonal display + accessories
    "CAM_FLOOR_B": {
        "ACCESSORIES":      (0.0,  0.0,  0.18, 0.7),    # Left wall column
        "SEASONAL_DISPLAY": (0.18, 0.55, 0.65, 1.0),    # Centre-bottom island
        "MAKEUP":           (0.18, 0.0,  1.0,  0.55),   # Makeup wall (upper)
    },

    # ── CAM_BILLING_01 (CAM_5.mp4) ───────────────────────────────────────────
    # POS counter — full frame is the billing zone
    # Customer standing in front of counter: right half of frame
    "CAM_BILLING_01": {
        "BILLING_COUNTER": (0.0, 0.0, 1.0, 1.0),
    },

    # ── CAM_STOCKROOM_01 (CAM_4.mp4) ─────────────────────────────────────────
    # Staff-only back room — full frame, all detections are staff
    "CAM_STOCKROOM_01": {
        "STOCKROOM": (0.0, 0.0, 1.0, 1.0),
    },
}

# Cameras where every person detected is definitionally staff
STAFF_ONLY_CAMERAS = {"CAM_STOCKROOM_01"}

# For entry camera: Y ratio below which a person is considered INBOUND (entering)
# In CAM_3, the door/foyer area is in the lower portion of the frame.
# Person appearing in bottom 55% → coming from outside = INBOUND
# Person in top 45% and moving toward bottom → OUTBOUND
ENTRY_INBOUND_Y_THRESHOLD = 0.55

# Cameras that can produce ENTRY/EXIT events
ENTRY_CAMERAS = {"CAM_ENTRY_01"}

# Cameras that partially overlap (for cross-camera deduplication signal)
OVERLAPPING_CAMERA_PAIRS = {
    ("CAM_FLOOR_A", "CAM_FLOOR_B"),  # Both cover parts of the main floor
}


class ZoneMapper:
    """
    Maps (cx, cy) pixel centroids to zone_ids based on which camera captured the frame.

    Key behaviours:
      - STOCKROOM camera: all detections → is_staff=True, zone=STOCKROOM
      - ENTRY camera: determines INBOUND vs OUTBOUND from Y position + velocity
      - FLOOR cameras: maps centroid to whichever zone region it falls into
      - BILLING camera: always BILLING_COUNTER, but staff vs customer still classified
    """

    def __init__(self, store_layout: dict, frame_w: int, frame_h: int, default_camera_id: str):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.default_camera_id = default_camera_id

        # Build pixel-space rectangles for all cameras
        self._px_zones: dict = {}
        for cam_id, zones in CAMERA_ZONE_MAP.items():
            self._px_zones[cam_id] = {}
            for zone_id, (x1r, y1r, x2r, y2r) in zones.items():
                self._px_zones[cam_id][zone_id] = (
                    int(x1r * frame_w),
                    int(y1r * frame_h),
                    int(x2r * frame_w),
                    int(y2r * frame_h),
                )

    def is_staff_only_camera(self, camera_id: Optional[str] = None) -> bool:
        """Returns True if this camera only ever sees staff (e.g. stockroom)."""
        cam = camera_id or self.default_camera_id
        return cam in STAFF_ONLY_CAMERAS

    def is_entry_camera(self, camera_id: Optional[str] = None) -> bool:
        cam = camera_id or self.default_camera_id
        return cam in ENTRY_CAMERAS

    def get_zone(self, cx: int, cy: int, camera_id: Optional[str] = None) -> Optional[str]:
        """Return zone_id for centroid position on the given camera, or None."""
        cam = camera_id or self.default_camera_id
        zones = self._px_zones.get(cam, {})
        for zone_id, (x1, y1, x2, y2) in zones.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone_id
        return None

    def get_entry_direction(self, cy: int, frame_h: int) -> str:
        """
        For CAM_ENTRY_01 (CAM_3.mp4):
        The camera is mounted looking down at the glass door from inside the store.
        The door/foyer (exterior side) is in the bottom portion of the frame.
        The store interior is in the top portion.

        A person appearing in the lower portion → entering (INBOUND).
        A person moving into the upper portion from lower → crossed threshold = entry event.
        A person in upper portion moving toward bottom → OUTBOUND.
        """
        threshold_y = int(ENTRY_INBOUND_Y_THRESHOLD * frame_h)
        return "INBOUND" if cy >= threshold_y else "OUTBOUND"

    def get_all_zones_for_camera(self, camera_id: str) -> list:
        return list(self._px_zones.get(camera_id, {}).keys())

    def cameras_overlap(self, cam_a: str, cam_b: str) -> bool:
        """Check if two cameras have known spatial overlap (for dedup warning)."""
        pair = tuple(sorted([cam_a, cam_b]))
        return pair in {tuple(sorted(p)) for p in OVERLAPPING_CAMERA_PAIRS}
