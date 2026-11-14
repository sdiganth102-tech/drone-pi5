#!/usr/bin/env python3
"""
Main Application: Real-Time Human Detection & Pythagorean Geolocation on Drone/Raspberry Pi.
"""

import os
import sys
import time
import argparse
import logging
import yaml
import cv2

from core.detector import DroneHumanDetector
from core.pythagoras_geolocator import PythagorasGeolocator
from core.telemetry import get_telemetry_provider, MockTelemetryProvider
from core.rescue_logger import RescueLogger
from utils.hud_visualizer import DroneHudVisualizer
from utils.video_stream import VideoStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DroneFloodRescue")

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using fallback defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description="Drone Flood Rescue: Human Detection & Geolocation System")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--source", type=str, default=None, help="Video source (0 for USB/PiCam, or path to video file)")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold (default 0.80 / 80%)")
    parser.add_argument("--alt", type=float, default=None, help="Drone altitude in meters")
    parser.add_argument("--tilt", type=float, default=None, help="Camera tilt/pitch downwards in degrees (0-90)")
    parser.add_argument("--lat", type=float, default=None, help="Drone GPS Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Drone GPS Longitude")
    parser.add_argument("--heading", type=float, default=None, help="Drone Heading in degrees (0-360)")
    parser.add_argument("--record", type=str, default=None, help="Output MP4 recording path")
    parser.add_argument("--no-display", action="store_true", help="Run headless without GUI window (for Pi flight)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.conf is not None:
        config.setdefault("model", {})["confidence_threshold"] = args.conf
    if args.alt is not None:
        config.setdefault("drone", {})["default_altitude_m"] = args.alt
    if args.lat is not None:
        config.setdefault("drone", {})["default_latitude"] = args.lat
    if args.lon is not None:
        config.setdefault("drone", {})["default_longitude"] = args.lon
    if args.heading is not None:
        config.setdefault("drone", {})["default_heading_deg"] = args.heading
    if args.tilt is not None:
        config.setdefault("camera", {})["tilt_pitch_deg"] = args.tilt

    model_cfg = config.get("model", {})
    drone_cfg = config.get("drone", {})
    cam_cfg = config.get("camera", {})
    out_cfg = config.get("output", {})

    video_source = args.source if args.source is not None else cam_cfg.get("input_source", 0)
    # Check if video source is integer (e.g. camera index)
    try:
        video_source = int(video_source)
    except ValueError:
        pass

    logger.info("=" * 65)
    logger.info("   DRONE FLOOD RESCUE SYSTEM - PYTHAGOREAN HUMAN GEOLOCATOR   ")
    logger.info("=" * 65)
    logger.info(f"Target Confidence Threshold : {model_cfg.get('confidence_threshold', 0.80) * 100:.1f}%")
    logger.info(f"Drone Reference Altitude    : {drone_cfg.get('default_altitude_m', 25.0)} m")
    logger.info(f"Camera Tilt Pitch Angle     : {cam_cfg.get('tilt_pitch_deg', 45.0)} deg")
    logger.info(f"Video Input Source          : {video_source}")
    logger.info("=" * 65)

    # 1. Initialize Detector
    detector = DroneHumanDetector(
        model_path=model_cfg.get("model_path", "models/yolov8n.pt"),
        confidence_threshold=model_cfg.get("confidence_threshold", 0.80),
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
        human_class_id=model_cfg.get("human_class_id", 0),
        inference_size=model_cfg.get("inference_size", 640),
        half_precision=model_cfg.get("half_precision", False),
        enable_tracking=model_cfg.get("enable_tracking", True)
    )

    # 2. Initialize Pythagoras Geolocator
    geolocator = PythagorasGeolocator(
        horizontal_fov_deg=cam_cfg.get("horizontal_fov_deg", 62.2),
        vertical_fov_deg=cam_cfg.get("vertical_fov_deg", 48.8),
        default_tilt_deg=cam_cfg.get("tilt_pitch_deg", 45.0)
    )

    # 3. Initialize Telemetry Provider
    telemetry = get_telemetry_provider(config)
    if isinstance(telemetry, MockTelemetryProvider):
        # Override initial telemetry values if provided via args
        if args.lat is not None: telemetry.latitude = args.lat
        if args.lon is not None: telemetry.longitude = args.lon
        if args.alt is not None: telemetry.altitude_m = args.alt
        if args.heading is not None: telemetry.heading_deg = args.heading
        if args.tilt is not None: telemetry.camera_tilt_deg = args.tilt

    # 4. Initialize Rescue Operations Logger
    logger_out = RescueLogger(
        output_dir=out_cfg.get("log_dir", "output_logs"),
        save_snapshots=out_cfg.get("save_snapshots", True),
        snapshot_cooldown_sec=out_cfg.get("snapshot_cooldown_sec", 2.0),
        export_csv=out_cfg.get("export_csv", True),
        export_geojson=out_cfg.get("export_geojson", True)
    )

    # 5. Initialize Video Stream
    stream = VideoStream(
        source=video_source,
        width=cam_cfg.get("frame_width", 640),
        height=cam_cfg.get("frame_height", 480)
    )
    stream.start()

    # 6. Initialize HUD Visualizer
    hud = DroneHudVisualizer(
        frame_width=cam_cfg.get("frame_width", 640),
        frame_height=cam_cfg.get("frame_height", 480)
    )

    # Optional Video Writer
    video_writer = None
    record_path = args.record
    if record_path:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)

    frame_count = 0
    t_start = time.time()
    fps_smooth = 0.0

    show_gui = not args.no_display
    if show_gui:
        window_title = "Drone Flood Rescue - Pythagorean Geolocation HUD"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    logger.info("System operational. Searching for humans in flood zone...")
    if show_gui:
        logger.info("Press 'q' to exit, 'SPACE' to pause.")

    is_paused = False

    try:
        while True:
            if not is_paused:
                ret, frame = stream.read()
                if not ret or frame is None:
                    # Video stream finished or waiting
                    if isinstance(video_source, str) and os.path.exists(video_source):
                        logger.info("Video file completed.")
                        break
                    time.sleep(0.01)
                    continue

                frame_count += 1
                t_frame_start = time.time()
                fh, fw = frame.shape[:2]

                if video_writer is None and record_path:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(record_path, fourcc, 25.0, (fw, fh))

                # Step A: Query Drone Telemetry (Alt, Lat, Lon, Hdg, Tilt)
                drone_state = telemetry.get_state()

                # Step B: Run Human Detector (Filtered for confidence >= 80%)
                detections = detector.detect(frame)

                # Step C: Pythagorean Geolocation for each verified human detection
                geo_results = []
                for det in detections:
                    geo = geolocator.locate_target(
                        drone_lat=drone_state.latitude,
                        drone_lon=drone_state.longitude,
                        drone_alt_m=drone_state.altitude_m,
                        drone_heading_deg=drone_state.heading_deg,
                        bbox=det["bbox"],
                        frame_width=fw,
                        frame_height=fh,
                        camera_tilt_deg=drone_state.camera_tilt_deg,
                        camera_yaw_offset_deg=cam_cfg.get("yaw_offset_deg", 0.0),
                        contact_point="bottom_center"
                    )
                    geo_results.append(geo)

                    # Step D: Log detection, save snapshot and export to CSV/GeoJSON
                    logger_out.log_detection(det, geo, frame)

                # FPS calculation
                dt = time.time() - t_frame_start
                instant_fps = (1.0 / dt) if dt > 0 else 30.0
                fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps

                # Step E: Render Tactical HUD Overlay
                annotated_frame = hud.render(
                    frame=frame,
                    drone_state=drone_state,
                    detections=detections,
                    geo_results=geo_results,
                    fps=fps_smooth
                )

                if video_writer:
                    video_writer.write(annotated_frame)

                if show_gui:
                    cv2.imshow(window_title, annotated_frame)

                if args.max_frames > 0 and frame_count >= args.max_frames:
                    logger.info(f"Reached max frames limit: {args.max_frames}")
                    break

            # Handle key events
            if show_gui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user.")
                    break
                elif key == ord(' '):
                    is_paused = not is_paused
                    logger.info("Paused." if is_paused else "Resumed.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stream.release()
        telemetry.close()
        if video_writer:
            video_writer.release()
        if show_gui:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/max(elapsed, 0.1):.1f} FPS avg).")
        logger.info(f"Rescue records logged to: {logger_out.csv_path}")
        logger.info(f"GIS GeoJSON saved to: {logger_out.geojson_path}")
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
