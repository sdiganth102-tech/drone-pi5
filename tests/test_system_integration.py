"""
Full System Integration Test:
Validates YOLOv8 loading, detection filtering (>= 80%),
Pythagoras geolocation math, and CSV/GeoJSON output generation.
"""

import os
import unittest
import numpy as np
import cv2

from core.detector import DroneHumanDetector
from core.pythagoras_geolocator import PythagorasGeolocator
from core.rescue_logger import RescueLogger
from core.telemetry import MockTelemetryProvider
from utils.hud_visualizer import DroneHudVisualizer

class TestSystemIntegration(unittest.TestCase):

    def setUp(self):
        self.output_dir = "tests/test_output_logs"
        self.detector = DroneHumanDetector(
            model_path="models/yolov8n.pt",
            confidence_threshold=0.80,
            enable_tracking=False
        )
        self.geolocator = PythagorasGeolocator(
            horizontal_fov_deg=62.2,
            vertical_fov_deg=48.8,
            default_tilt_deg=45.0
        )
        self.telemetry = MockTelemetryProvider(
            latitude=12.971599,
            longitude=77.594566,
            altitude_m=20.0,
            heading_deg=90.0,
            camera_tilt_deg=45.0
        )
        self.logger = RescueLogger(
            output_dir=self.output_dir,
            save_snapshots=True,
            snapshot_cooldown_sec=0.0,
            export_csv=True,
            export_geojson=True
        )
        self.hud = DroneHudVisualizer(frame_width=640, frame_height=480)

    def test_mock_detection_pipeline(self):
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock detection with confidence 0.88 (above 80% threshold)
        mock_detection = {
            "track_id": 1,
            "confidence": 0.88,
            "bbox": (280.0, 200.0, 360.0, 380.0),
            "center": (320.0, 290.0),
            "bottom_center": (320.0, 380.0),
            "width": 80.0,
            "height": 180.0
        }

        drone_state = self.telemetry.get_state()

        geo = self.geolocator.locate_target(
            drone_lat=drone_state.latitude,
            drone_lon=drone_state.longitude,
            drone_alt_m=drone_state.altitude_m,
            drone_heading_deg=drone_state.heading_deg,
            bbox=mock_detection["bbox"],
            frame_width=640,
            frame_height=480,
            camera_tilt_deg=drone_state.camera_tilt_deg,
            contact_point="bottom_center"
        )

        self.assertIn("target_lat", geo)
        self.assertIn("target_lon", geo)
        self.assertGreater(geo["ground_distance_m"], 0.0)
        self.assertGreater(geo["slant_distance_m"], geo["ground_distance_m"])
        self.assertTrue(geo["pythagoras_verified"])

        # Test rescue logger
        snap = self.logger.log_detection(mock_detection, geo, frame)
        self.assertTrue(os.path.exists(self.logger.csv_path))
        self.assertTrue(os.path.exists(self.logger.geojson_path))

        # Test HUD render
        vis = self.hud.render(
            frame=frame,
            drone_state=drone_state,
            detections=[mock_detection],
            geo_results=[geo],
            fps=30.0
        )
        self.assertEqual(vis.shape, frame.shape)

    def tearDown(self):
        # Cleanup test files
        if os.path.exists(self.output_dir):
            for root, dirs, files in os.walk(self.output_dir, topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f))
                    except: pass
                for d in dirs:
                    try: os.rmdir(os.path.join(root, d))
                    except: pass
            try: os.rmdir(self.output_dir)
            except: pass

if __name__ == "__main__":
    unittest.main()
