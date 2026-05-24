"""
zone_mapper.py — Maps pixel coordinates to named store zones.

Zones are defined as rectangular regions proportional to frame dimensions.
In production these would come from calibration overlaid on store_layout.json.
For this implementation, zones are defined as normalised [x1, y1, x2, y2] ratios.

Camera-specific zone definitions:
  - CAM_ENTRY_01: Entry/exit threshold detection + direction
  - CAM_FLOOR_01: Product zone detection
  - CAM_BILLING_01: Billing counter + queue region
"""

from typing import Optional


# Normalised zone definitions per camera type: [x1_ratio, y1_ratio, x2_ratio, y2_ratio]
# These represent the fraction of frame width/height that each zone occupies.
# In production: calibrated from store blueprint overlay on camera feed.

CAMERA_ZONE_MAP = {
    "CAM_ENTRY_01": {
        "ENTRY_THRESHOLD": [0.0, 0.6, 1.0, 1.0],   # Bottom 40% of frame (door area)
    },
    "CAM_FLOOR_01": {
        "SKINCARE":      [0.0,  0.0, 0.5,  0.45],  # Top-left quadrant
        "HAIRCARE":      [0.5,  0.0, 1.0,  0.45],  # Top-right quadrant
        "FRAGRANCE":     [0.0,  0.45, 0.5, 0.75],  # Mid-left
        "PERSONAL_CARE": [0.5,  0.45, 1.0, 0.75],  # Mid-right
    },
    "CAM_BILLING_01": {
        "BILLING_COUNTER": [0.0, 0.3, 1.0, 1.0],   # Lower 70% — counter + queue area
    },
}

# Entry direction: if centroid Y is in the bottom portion → entering (INBOUND)
# if centroid Y is in the top portion → exiting (OUTBOUND)
ENTRY_THRESHOLD_Y_RATIO = 0.6   # Below this = INBOUND


class ZoneMapper:
    """
    Maps (cx, cy) pixel coordinates to a zone_id given the camera context.

    Usage:
        mapper = ZoneMapper(store_layout, frame_w, frame_h, camera_id)
        zone = mapper.get_zone(cx, cy, camera_id)
        direction = mapper.get_entry_direction(cy, frame_h)
    """

    def __init__(self, store_layout: dict, frame_w: int, frame_h: int, default_camera_id: str):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.default_camera_id = default_camera_id

        # Build pixel-space zone rectangles
        self.camera_zones: dict = {}
        for cam_type, zones in CAMERA_ZONE_MAP.items():
            self.camera_zones[cam_type] = {}
            for zone_id, (x1r, y1r, x2r, y2r) in zones.items():
                self.camera_zones[cam_type][zone_id] = (
                    int(x1r * frame_w),
                    int(y1r * frame_h),
                    int(x2r * frame_w),
                    int(y2r * frame_h),
                )

    def get_zone(self, cx: int, cy: int, camera_id: Optional[str] = None) -> Optional[str]:
        """Return zone_id for given centroid, or None if not in any zone."""
        cam = camera_id or self.default_camera_id
        zones = self.camera_zones.get(cam, {})

        for zone_id, (x1, y1, x2, y2) in zones.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone_id
        return None

    def get_entry_direction(self, cy: int, frame_h: int) -> str:
        """
        For entry camera: determine if movement is inbound or outbound.
        People appearing in bottom region of entry camera frame → entering store.
        People appearing in top region → exiting store.
        """
        threshold_y = int(ENTRY_THRESHOLD_Y_RATIO * frame_h)
        if cy >= threshold_y:
            return "INBOUND"
        else:
            return "OUTBOUND"

    def get_all_zones_for_camera(self, camera_id: str) -> list:
        return list(self.camera_zones.get(camera_id, {}).keys())
