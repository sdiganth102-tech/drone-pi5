"""
Rescue Operation Logger and GIS Exporter.

Logs high-confidence human detections (>= 80%) with exact GPS coordinates,
saves image snapshots with telemetry overlays, and exports rescue data to
standard CSV and GeoJSON formats for disaster response coordination.
"""

import os
import csv
import json
import time
import logging
from typing import Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class RescueLogger:
    """
    Handles logging of verified human detections, saving visual proof snapshots,
    and maintaining GIS-compatible records for flood rescue teams.
    """
    def __init__(
        self,
        output_dir: str = "output_logs",
        save_snapshots: bool = True,
        snapshot_cooldown_sec: float = 2.0,
        export_csv: bool = True,
        export_geojson: bool = True
    ):
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.save_snapshots = save_snapshots
        self.snapshot_cooldown_sec = snapshot_cooldown_sec
        self.export_csv = export_csv
        self.export_geojson = export_geojson

        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_snapshots:
            os.makedirs(self.snapshots_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "rescue_detections.csv")
        self.geojson_path = os.path.join(output_dir, "rescue_targets.geojson")

        self.last_snapshot_time: Dict[Any, float] = {}
        self.records = []

        if self.export_csv and not os.path.exists(self.csv_path):
            self._init_csv()

    def _init_csv(self):
        headers = [
            "timestamp_iso",
            "track_id",
            "confidence_pct",
            "target_latitude",
            "target_longitude",
            "ground_distance_m",
            "slant_distance_m",
            "target_bearing_deg",
            "drone_latitude",
            "drone_longitude",
            "drone_altitude_m",
            "drone_heading_deg",
            "camera_tilt_deg",
            "pythagoras_verified",
            "snapshot_file"
        ]
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_detection(
        self,
        detection: Dict[str, Any],
        geo_info: Dict[str, Any],
        frame: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Record a verified human detection.

        Args:
            detection: Detector output dictionary (conf, bbox, track_id, etc.)
            geo_info: PythagorasGeolocator output dictionary
            frame: Full frame image for snapshot cropping and saving.

        Returns:
            Snapshot filename if saved, else None.
        """
        now = time.time()
        track_id = detection.get("track_id")
        conf_pct = round(detection["confidence"] * 100.0, 1)
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

        # Snapshot rate limiting
        key = track_id if track_id is not None else "generic"
        last_time = self.last_snapshot_time.get(key, 0.0)
        should_save_snap = self.save_snapshots and (now - last_time >= self.snapshot_cooldown_sec)

        snapshot_filename = ""
        if should_save_snap and frame is not None:
            self.last_snapshot_time[key] = now
            snapshot_filename = self._save_snapshot(detection, geo_info, frame, now)

        # Log entry
        record = {
            "timestamp_iso": iso_time,
            "track_id": track_id if track_id is not None else "N/A",
            "confidence_pct": conf_pct,
            "target_latitude": geo_info["target_lat"],
            "target_longitude": geo_info["target_lon"],
            "ground_distance_m": geo_info["ground_distance_m"],
            "slant_distance_m": geo_info["slant_distance_m"],
            "target_bearing_deg": geo_info["target_bearing_deg"],
            "drone_latitude": geo_info["drone_lat"],
            "drone_longitude": geo_info["drone_lon"],
            "drone_altitude_m": geo_info["drone_alt_m"],
            "drone_heading_deg": geo_info.get("drone_heading_deg", 0.0),
            "camera_tilt_deg": geo_info["depression_angle_deg"],
            "pythagoras_verified": geo_info.get("pythagoras_verified", True),
            "snapshot_file": snapshot_filename
        }
        self.records.append(record)

        # Write to CSV
        if self.export_csv:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["timestamp_iso"],
                    record["track_id"],
                    record["confidence_pct"],
                    record["target_latitude"],
                    record["target_longitude"],
                    record["ground_distance_m"],
                    record["slant_distance_m"],
                    record["target_bearing_deg"],
                    record["drone_latitude"],
                    record["drone_longitude"],
                    record["drone_altitude_m"],
                    record["drone_heading_deg"],
                    record["camera_tilt_deg"],
                    record["pythagoras_verified"],
                    record["snapshot_file"]
                ])

        # Write to GeoJSON
        if self.export_geojson:
            self._update_geojson()

        logger.info(
            f"[RESCUE ALERT] Human #{record['track_id']} at {record['target_latitude']:.6f}, {record['target_longitude']:.6f} "
            f"| Dist: {record['ground_distance_m']}m (Slant: {record['slant_distance_m']}m) | Conf: {conf_pct}%"
        )

        return snapshot_filename or None

    def _save_snapshot(
        self,
        detection: Dict[str, Any],
        geo_info: Dict[str, Any],
        frame: np.ndarray,
        timestamp: float
    ) -> str:
        tid = detection.get("track_id", "untracked")
        filename = f"rescue_survivor_ID{tid}_{int(timestamp)}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        snap = frame.copy()
        x1, y1, x2, y2 = map(int, detection["bbox"])

        # Draw red alert bounding box
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Telemetry banner at bottom
        h, w = snap.shape[:2]
        banner_h = 45
        cv2.rectangle(snap, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        
        text1 = (f"HUMAN LOCATED: {geo_info['target_lat']:.6f}, {geo_info['target_lon']:.6f} | "
                 f"Ground Dist: {geo_info['ground_distance_m']}m | Slant: {geo_info['slant_distance_m']}m")
        text2 = f"Drone Alt: {geo_info['drone_alt_m']}m | Tilt: {geo_info['depression_angle_deg']} deg | Conf: {detection['confidence']*100:.1f}%"
        
        cv2.putText(snap, text1, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(snap, text2, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(filepath, snap)
        return filename

    def _update_geojson(self):
        features = []
        for r in self.records:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["target_longitude"], r["target_latitude"]]
                },
                "properties": {
                    "track_id": r["track_id"],
                    "confidence_pct": r["confidence_pct"],
                    "ground_distance_m": r["ground_distance_m"],
                    "slant_distance_m": r["slant_distance_m"],
                    "timestamp": r["timestamp_iso"],
                    "drone_alt_m": r["drone_altitude_m"],
                    "snapshot": r["snapshot_file"]
                }
            }
            features.append(feat)

        geojson_doc = {
            "type": "FeatureCollection",
            "name": "DroneFloodRescueTargets",
            "features": features
        }

        with open(self.geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)
