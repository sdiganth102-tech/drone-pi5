#!/usr/bin/env python3
"""
Drone Flood Rescue: Web Mission Control Station.
Runs a localhost web application allowing users to select flood drone videos,
tune flight altitude & camera tilt, and stream real-time Pythagorean detections.
"""

import os
import sys
import time
import math
import json
import logging
import threading
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

# Project modules
from core.detector import DroneHumanDetector
from core.pythagoras_geolocator import PythagorasGeolocator
from core.telemetry import MockTelemetryProvider
from core.rescue_logger import RescueLogger
from utils.hud_visualizer import DroneHudVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebMissionControl")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "test_videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "output_logs")
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")
THUMBNAILS_DIR = os.path.join(BASE_DIR, "static", "thumbnails")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Video metadata dictionary
VIDEO_METADATA = {
    "people-detection.mp4": {
        "title": "Flood Walking Survivor Search",
        "description": "High-confidence detection benchmark with walking survivors in disaster zone.",
        "badge": "Recommended Demo",
        "difficulty": "High Precision"
    },
    "clip1_drone_drowning.mp4": {
        "title": "Drowning Survivor Emergency",
        "description": "Aerial drone surveillance over rapid water current with drowning victim.",
        "badge": "Flood Water",
        "difficulty": "Lifeguard Ops"
    },
    "clip2_flood_village_drones.mp4": {
        "title": "Submerged Village Survey",
        "description": "Drone flyover surveying flooded residential settlements and stranded villagers.",
        "badge": "Flood Disaster",
        "difficulty": "Aerial View"
    },
    "clip4_flood_stranded_river.mp4": {
        "title": "Stranded by River Flood",
        "description": "Survivors isolated on small elevated terrain surrounded by flash flood.",
        "badge": "Flash Flood",
        "difficulty": "Rescue Priority"
    },
    "clip5_drone_isolated_rescue.mp4": {
        "title": "Isolated Survivor Location",
        "description": "Remote survivor stranded in rugged flood basin awaiting extraction.",
        "badge": "Remote Search",
        "difficulty": "Search & Locate"
    },
    "clip3_lifeguard_drone_drowning.mp4": {
        "title": "Lifeguard Drone Deployment",
        "description": "Emergency responder drone targeting distressed swimmer in water.",
        "badge": "Emergency",
        "difficulty": "Dynamic Water"
    },
    "pedestrians.mp4": {
        "title": "Urban Flood Evacuation",
        "description": "Multiple people navigating inundated urban street environment.",
        "badge": "Urban Search",
        "difficulty": "Multi-Target"
    },
    "multi_people.mp4": {
        "title": "Stranded Group Tracking",
        "description": "Tracking persistent IDs across multiple individuals in open terrain.",
        "badge": "Group Tracking",
        "difficulty": "Dense Cluster"
    }
}

class MissionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.is_paused = False
        self.active_video = "people-detection.mp4"
        self.drone_alt = 25.0
        self.camera_tilt = 45.0
        self.conf_threshold = 0.80
        self.drone_lat = 12.971599
        self.drone_lon = 77.594566
        self.drone_heading = 45.0

        self.status = "STANDBY"
        self.survivor_count = 0
        self.primary_target = {
            "lat": 0.0,
            "lon": 0.0,
            "ground_distance_m": 0.0,
            "slant_distance_m": 0.0,
            "target_bearing_deg": 0.0,
            "confidence_pct": 0.0,
            "track_id": None
        }
        self.recent_events = []
        self.current_frame = None

state = MissionState()

# Global worker pipeline references
worker_thread: Optional[threading.Thread] = None

def get_available_videos():
    videos = []
    if os.path.exists(VIDEOS_DIR):
        files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for f in files:
            meta = VIDEO_METADATA.get(f, {
                "title": f.replace("_", " ").replace(".mp4", "").title(),
                "description": "Drone aerial reconnaissance video clip.",
                "badge": "Video",
                "difficulty": "Standard"
            })
            thumb = f"{f}.jpg" if os.path.exists(os.path.join(THUMBNAILS_DIR, f"{f}.jpg")) else "default.jpg"
            videos.append({
                "filename": f,
                "title": meta["title"],
                "description": meta["description"],
                "badge": meta["badge"],
                "difficulty": meta["difficulty"],
                "thumbnail": f"/static/thumbnails/{thumb}"
            })
    return videos

def mission_pipeline_worker():
    global state
    logger.info("Mission pipeline worker started.")

    try:
        model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
        detector = DroneHumanDetector(
            model_path=model_path,
            confidence_threshold=state.conf_threshold,
            enable_tracking=True
        )

        geolocator = PythagorasGeolocator(
            horizontal_fov_deg=62.2,
            vertical_fov_deg=48.8,
            default_tilt_deg=state.camera_tilt
        )

        telemetry = MockTelemetryProvider(
            latitude=state.drone_lat,
            longitude=state.drone_lon,
            altitude_m=state.drone_alt,
            heading_deg=state.drone_heading,
            camera_tilt_deg=state.camera_tilt,
            simulate_drift=True
        )

        rescue_logger = RescueLogger(
            output_dir=LOGS_DIR,
            save_snapshots=True,
            snapshot_cooldown_sec=1.5,
            export_csv=True,
            export_geojson=True
        )

        hud = DroneHudVisualizer(frame_width=640, frame_height=480)

        video_path = os.path.join(VIDEOS_DIR, state.active_video)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            with state.lock:
                state.is_running = False
                state.status = "ERROR: Cannot open video"
            return

        with state.lock:
            state.status = "SCANNING FLOOD ZONE"

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / max(10.0, min(60.0, fps))

        while state.is_running:
            if state.is_paused:
                time.sleep(0.05)
                continue

            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            fh, fw = frame.shape[:2]

            # Dynamic parameters sync
            telemetry.altitude_m = state.drone_alt
            telemetry.camera_tilt_deg = state.camera_tilt
            detector.confidence_threshold = state.conf_threshold

            drone_state = telemetry.get_state()
            detections = detector.detect(frame)

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
                    contact_point="bottom_center"
                )
                geo_results.append(geo)
                snap = rescue_logger.log_detection(det, geo, frame)

                if snap:
                    with state.lock:
                        state.recent_events.insert(0, {
                            "time": time.strftime("%H:%M:%S"),
                            "track_id": det.get("track_id", "1"),
                            "conf": f"{det['confidence']*100:.1f}%",
                            "lat": round(geo["target_lat"], 6),
                            "lon": round(geo["target_lon"], 6),
                            "ground_dist": f"{geo['ground_distance_m']}m",
                            "slant_dist": f"{geo['slant_distance_m']}m",
                            "bearing": f"{geo['target_bearing_deg']}°",
                            "snapshot": snap
                        })
                        if len(state.recent_events) > 30:
                            state.recent_events.pop()

            # Render HUD
            annotated = hud.render(
                frame=frame,
                drone_state=drone_state,
                detections=detections,
                geo_results=geo_results,
                fps=fps
            )

            # Update shared state
            with state.lock:
                state.current_frame = annotated
                state.survivor_count = len(detections)
                if len(detections) > 0:
                    state.status = f"SURVIVOR DETECTED ({len(detections)})"
                    p_geo = geo_results[0]
                    p_det = detections[0]
                    state.primary_target = {
                        "lat": round(p_geo["target_lat"], 6),
                        "lon": round(p_geo["target_lon"], 6),
                        "ground_distance_m": p_geo["ground_distance_m"],
                        "slant_distance_m": p_geo["slant_distance_m"],
                        "target_bearing_deg": p_geo["target_bearing_deg"],
                        "confidence_pct": round(p_det["confidence"] * 100, 1),
                        "track_id": p_det.get("track_id", "1")
                    }
                else:
                    if not state.is_paused:
                        state.status = "SCANNING FLOOD ZONE"

            elapsed = time.time() - t_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)

        cap.release()

    except Exception as e:
        logger.error(f"Worker exception: {e}")
        with state.lock:
            state.status = f"ERROR: {str(e)}"
    finally:
        with state.lock:
            state.is_running = False
        logger.info("Mission pipeline worker stopped.")

def generate_video_stream():
    """Generates MJPEG multipart stream for HTTP video element."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "SYSTEM STANDBY - CLICK 'START MISSION'", (70, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    _, blank_jpeg = cv2.imencode('.jpg', blank_frame)
    blank_bytes = blank_jpeg.tobytes()

    while True:
        frame = None
        with state.lock:
            if state.current_frame is not None and state.is_running:
                frame = state.current_frame.copy()

        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = jpeg.tobytes()
            else:
                frame_bytes = blank_bytes
        else:
            frame_bytes = blank_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS stream

# =====================
# ROUTES
# =====================

@app.route("/")
def index():
    videos = get_available_videos()
    return render_template("index.html", videos=videos, default_video=state.active_video)

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/videos")
def api_videos():
    return jsonify(get_available_videos())

@app.route("/api/start", methods=["POST"])
def api_start():
    global worker_thread, state
    data = request.get_json() or {}

    video_name = data.get("video_name", state.active_video)
    alt = float(data.get("altitude", state.drone_alt))
    tilt = float(data.get("tilt", state.camera_tilt))
    conf = float(data.get("confidence", state.conf_threshold))

    with state.lock:
        state.active_video = video_name
        state.drone_alt = alt
        state.camera_tilt = tilt
        state.conf_threshold = conf
        state.is_paused = False

        if not state.is_running:
            state.is_running = True
            worker_thread = threading.Thread(target=mission_pipeline_worker, daemon=True)
            worker_thread.start()

    return jsonify({"success": True, "status": "Mission started", "video": video_name})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    with state.lock:
        if state.is_running:
            state.is_paused = not state.is_paused
            status_text = "PAUSED" if state.is_paused else "RESUMED"
            return jsonify({"success": True, "is_paused": state.is_paused, "status": status_text})
    return jsonify({"success": False, "message": "Mission not running"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state.lock:
        state.is_running = False
        state.is_paused = False
        state.status = "MISSION STOPPED"
        state.survivor_count = 0
    return jsonify({"success": True, "status": "Mission stopped"})

@app.route("/api/update_params", methods=["POST"])
def api_update_params():
    data = request.get_json() or {}
    with state.lock:
        if "altitude" in data: state.drone_alt = float(data["altitude"])
        if "tilt" in data: state.camera_tilt = float(data["tilt"])
        if "confidence" in data: state.conf_threshold = float(data["confidence"])
    return jsonify({"success": True, "altitude": state.drone_alt, "tilt": state.camera_tilt, "confidence": state.conf_threshold})

@app.route("/api/telemetry")
def api_telemetry():
    with state.lock:
        return jsonify({
            "status": state.status,
            "is_running": state.is_running,
            "is_paused": state.is_paused,
            "active_video": state.active_video,
            "altitude": state.drone_alt,
            "tilt": state.camera_tilt,
            "confidence": state.conf_threshold,
            "survivor_count": state.survivor_count,
            "primary_target": state.primary_target,
            "recent_events": state.recent_events[:15]
        })

@app.route("/snapshots/<path:filename>")
def get_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route("/api/download/csv")
def download_csv():
    csv_file = os.path.join(LOGS_DIR, "rescue_detections.csv")
    if os.path.exists(csv_file):
        return send_from_directory(LOGS_DIR, "rescue_detections.csv", as_attachment=True)
    return jsonify({"error": "No CSV records found yet"}), 404

@app.route("/api/download/geojson")
def download_geojson():
    geo_file = os.path.join(LOGS_DIR, "rescue_targets.geojson")
    if os.path.exists(geo_file):
        return send_from_directory(LOGS_DIR, "rescue_targets.geojson", as_attachment=True)
    return jsonify({"error": "No GeoJSON records found yet"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Drone Flood Rescue Web Server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
