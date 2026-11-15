#!/usr/bin/env python3
"""
Simulate Flood Search & Rescue Mission.

Runs automated flight mission over flood disaster video clips,
validating >= 80% human detection, Pythagorean distance estimation,
and geodetic coordinate generation.
"""

import os
import sys
import time
import argparse
import logging
import cv2

from core.detector import DroneHumanDetector
from core.pythagoras_geolocator import PythagorasGeolocator
from core.telemetry import MockTelemetryProvider
from core.rescue_logger import RescueLogger
from utils.hud_visualizer import DroneHudVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FloodSimulation")

def run_mission_simulation(
    video_path: str,
    output_video_path: str,
    drone_alt_m: float = 20.0,
    drone_lat: float = 12.971599,
    drone_lon: float = 77.594566,
    drone_heading: float = 45.0,
    camera_tilt_deg: float = 45.0,
    max_frames: int = 200,
    conf_threshold: float = 0.80,
    show_window: bool = False
):
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    logger.info("=" * 60)
    logger.info(f"SIMULATING FLOOD MISSION ON: {os.path.basename(video_path)}")
    logger.info(f"Drone Altitude: {drone_alt_m}m | Camera Tilt: {camera_tilt_deg}° | Conf Threshold: {conf_threshold*100:.0f}%")
    logger.info("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = DroneHumanDetector(
        model_path="models/yolov8n.pt",
        confidence_threshold=conf_threshold,
        enable_tracking=True
    )

    geolocator = PythagorasGeolocator(
        horizontal_fov_deg=62.2,
        vertical_fov_deg=48.8,
        default_tilt_deg=camera_tilt_deg
    )

    telemetry = MockTelemetryProvider(
        latitude=drone_lat,
        longitude=drone_lon,
        altitude_m=drone_alt_m,
        heading_deg=drone_heading,
        camera_tilt_deg=camera_tilt_deg,
        simulate_drift=True
    )

    rescue_logger = RescueLogger(
        output_dir="output_logs",
        save_snapshots=True,
        snapshot_cooldown_sec=1.5,
        export_csv=True,
        export_geojson=True
    )

    hud = DroneHudVisualizer(frame_width=fw, frame_height=fh)

    if show_window:
        cv2.namedWindow("Flood Rescue Mission Simulation", cv2.WINDOW_NORMAL)
        logger.info("Live display window OPENED. Press 'q' to stop, 'SPACE' to pause.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (fw, fh))

    frame_idx = 0
    total_detections_80plus = 0
    detected_survivor_records = []

    t_start = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        drone_state = telemetry.get_state()

        # Run detector with >= 80% confidence
        detections = detector.detect(frame)

        geo_results = []
        for det in detections:
            total_detections_80plus += 1
            geo = geolocator.locate_target(
                drone_lat=drone_state.latitude,
                drone_lon=drone_state.longitude,
                drone_alt_m=drone_state.altitude_m,
                drone_heading_deg=drone_state.heading_deg,
                bbox=det["bbox"],
                frame_width=fw,
                frame_height=fh,
                camera_tilt_deg=drone_state.camera_tilt_deg,
                contact_point="bottom_center"
            )
            geo_results.append(geo)
            snap = rescue_logger.log_detection(det, geo, frame)
            if snap:
                detected_survivor_records.append({
                    "frame": frame_idx,
                    "track_id": det.get("track_id"),
                    "conf": f"{det['confidence']*100:.1f}%",
                    "lat": geo["target_lat"],
                    "lon": geo["target_lon"],
                    "ground_dist": f"{geo['ground_distance_m']}m",
                    "slant_dist": f"{geo['slant_distance_m']}m",
                    "snapshot": snap
                })

        # Render HUD
        annotated = hud.render(
            frame=frame,
            drone_state=drone_state,
            detections=detections,
            geo_results=geo_results,
            fps=fps
        )

        out.write(annotated)

        if show_window:
            cv2.imshow("Flood Rescue Mission Simulation", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Simulation stopped by user.")
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()
    elapsed = time.time() - t_start

    logger.info("\n" + "=" * 60)
    logger.info("MISSION SIMULATION SUMMARY:")
    logger.info(f"Frames processed          : {frame_idx} in {elapsed:.2f}s ({frame_idx/max(elapsed, 0.1):.1f} FPS)")
    logger.info(f"High-conf detections (>=80%): {total_detections_80plus}")
    logger.info(f"Unique survivor snapshots : {len(detected_survivor_records)}")
    logger.info(f"Annotated video saved to  : {output_video_path}")
    logger.info(f"Rescue log CSV            : {rescue_logger.csv_path}")
    logger.info(f"GeoJSON output            : {rescue_logger.geojson_path}")
    logger.info("=" * 60)

    for rec in detected_survivor_records[:5]:
        logger.info(f"Target #{rec['track_id']} (Conf: {rec['conf']}) at GPS: {rec['lat']:.6f}, {rec['lon']:.6f} | Dist: {rec['ground_dist']} (Slant: {rec['slant_dist']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate flood search mission")
    parser.add_argument("--video", type=str, default="test_videos/clip1_drone_drowning.mp4")
    parser.add_argument("--output", type=str, default="output_logs/simulated_mission_output.mp4")
    parser.add_argument("--alt", type=float, default=20.0)
    parser.add_argument("--tilt", type=float, default=45.0)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--conf", type=float, default=0.80)
    parser.add_argument("--show", action="store_true", help="Display live video window with tactical HUD during simulation")
    args = parser.parse_args()

    run_mission_simulation(
        video_path=args.video,
        output_video_path=args.output,
        drone_alt_m=args.alt,
        camera_tilt_deg=args.tilt,
        max_frames=args.frames,
        conf_threshold=args.conf,
        show_window=args.show
    )
