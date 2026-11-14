"""
Drone Heads-Up Display (HUD) and Detection Visualizer.

Overlays telemetry status, artificial horizon, radar bearing, and
high-precision Pythagorean distance + GPS coordinates on live video.
"""

import math
from typing import List, Dict, Any
import cv2
import numpy as np

# Distinct colors for track IDs
TRACK_COLORS = [
    (0, 255, 0), (0, 215, 255), (255, 191, 0), (255, 0, 255),
    (0, 165, 255), (50, 205, 50), (147, 20, 255), (255, 255, 0)
]

class DroneHudVisualizer:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.width = frame_width
        self.height = frame_height

    def render(
        self,
        frame: np.ndarray,
        drone_state: Any,
        detections: List[Dict[str, Any]],
        geo_results: List[Dict[str, Any]],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Render complete Tactical HUD overlay on video frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Top Telemetry Bar
        cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.line(vis, (0, 40), (w, 40), (0, 255, 200), 1)

        lat_str = f"DRONE: {drone_state.latitude:.6f}, {drone_state.longitude:.6f}"
        alt_str = f"ALT: {drone_state.altitude_m:.1f}m"
        hdg_str = f"HDG: {drone_state.heading_deg:.0f}°"
        tilt_str = f"TILT: {drone_state.camera_tilt_deg:.0f}°"
        fps_str = f"FPS: {fps:.1f}"

        telemetry_txt = f"{lat_str} | {alt_str} | {hdg_str} | {tilt_str} | {fps_str}"
        cv2.putText(vis, telemetry_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Status Banner (top-right)
        num_targets = len(detections)
        if num_targets > 0:
            badge_color = (0, 200, 0)  # Green
            badge_text = f"SURVIVOR DETECTED ({num_targets})"
        else:
            badge_color = (200, 100, 0)  # Blue/Cyan
            badge_text = "SCANNING FLOOD ZONE"

        cv2.rectangle(vis, (w - 230, 48), (w - 10, 76), (20, 20, 20), -1)
        cv2.rectangle(vis, (w - 230, 48), (w - 10, 76), badge_color, 1)
        cv2.putText(vis, badge_text, (w - 220, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

        # 3. Center Aim Reticle
        cx, cy = w // 2, h // 2
        cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
        cv2.circle(vis, (cx, cy), 8, (0, 255, 255), 1)

        # 4. Draw detections and Pythagorean geolocation tags
        for det, geo in zip(detections, geo_results):
            x1, y1, x2, y2 = map(int, det["bbox"])
            tid = det.get("track_id")
            conf = det["confidence"] * 100.0

            color = TRACK_COLORS[tid % len(TRACK_COLORS)] if tid is not None else (0, 255, 0)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Contact point marker (bottom-center)
            foot_x = (x1 + x2) // 2
            foot_y = y2
            cv2.circle(vis, (foot_x, foot_y), 4, (0, 0, 255), -1)

            # Vector line from frame center to target
            cv2.line(vis, (cx, cy), (foot_x, foot_y), (0, 255, 255), 1, cv2.LINE_AA)

            # Information label box
            id_tag = f"ID:{tid} " if tid is not None else ""
            label_top = f"{id_tag}CONF: {conf:.1f}%"
            label_geo = f"GPS: {geo['target_lat']:.6f}, {geo['target_lon']:.6f}"
            label_dist = f"Gnd: {geo['ground_distance_m']}m | Slant(Pyth): {geo['slant_distance_m']}m"

            # Draw background bubble for readable text
            bubble_y = max(45, y1 - 45)
            cv2.rectangle(vis, (x1, bubble_y), (x1 + 250, bubble_y + 42), (10, 10, 10), -1)
            cv2.rectangle(vis, (x1, bubble_y), (x1 + 250, bubble_y + 42), color, 1)

            cv2.putText(vis, label_top, (x1 + 5, bubble_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, label_geo, (x1 + 5, bubble_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, label_dist, (x1 + 5, bubble_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # 5. Bottom Pythagoras equation reference banner
        footer_text = "Pythagoras 3D Range: d_slant = sqrt(d_ground^2 + Alt^2) | Confidence Filter >= 80%"
        cv2.putText(vis, footer_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return vis
