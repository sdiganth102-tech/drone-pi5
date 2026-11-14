# Drone Flood Rescue: Human Detection & Pythagorean Geolocation System

Autonomous search-and-rescue payload software designed for **Raspberry Pi** mounted on a search-and-rescue drone. The system detects stranded humans in flood disaster zones with high confidence ($\ge 80\%$) and computes their exact geographic coordinates (Latitude, Longitude) and distance using **Pythagorean theorem** and **camera gimbal geometry**.

---

## 1. System Architecture & Mathematical Foundation

```
                           [ DRONE ]
                           /|      \
                          / |       \
                         /  |        \
      Line-of-Sight     /   |         \
      Slant Range      /    | Altitude \
      (d_slant)       /     |    (h)    \
                     /      |            \
                    /       |             \
                   /        |              \
                  /         |               \
                 v          v                v
           [SURVIVOR] <------------> [NADIR GROUND POINT]
                      Ground Distance
                         (d_ground)
```

### A. Pythagorean Theorem Application
In the right-angled triangle formed by:
1. **Vertical leg**: Drone altitude $h$ above the flood water / ground surface (obtained from barometer, LiDAR, or GPS relative altitude).
2. **Horizontal leg**: Ground distance $d_{ground}$ from the sub-drone nadir point to the survivor.
3. **Hypotenuse**: Line-of-sight slant distance $d_{slant}$ from the drone camera lens to the detected human.

By the **Pythagorean theorem**:
$$d_{slant}^2 = d_{ground}^2 + h^2 \implies d_{slant} = \sqrt{d_{ground}^2 + h^2}$$
$$d_{ground} = \sqrt{d_{slant}^2 - h^2}$$

### B. Camera Angle & Ray Trigonometry
Let the camera have horizontal Field of View $HFOV$, vertical Field of View $VFOV$, and frame dimensions $(W, H)$:
$$f_x = \frac{W / 2}{\tan(HFOV / 2)}, \quad f_y = \frac{H / 2}{\tan(VFOV / 2)}$$

For a detected human bounding box $[x_1, y_1, x_2, y_2]$, the ground contact point $(u, v)$ is the bottom-center $( \frac{x_1 + x_2}{2}, y_2 )$ (where survivor contacts the water or surface):
- **Azimuth pixel angle offset**: $\Delta \alpha_x = \arctan\left(\frac{u - W/2}{f_x}\right)$
- **Elevation pixel angle offset**: $\Delta \alpha_y = \arctan\left(\frac{v - H/2}{f_y}\right)$

With camera gimbal tilt $\theta_{tilt}$ (depression angle downwards from horizontal):
- **Total depression angle below horizon**: $\gamma = \theta_{tilt} + \Delta \alpha_y$
- **Horizontal Ground Distance**: $d_{ground} = \frac{h}{\tan(\gamma)}$
- **Target True Bearing**: $\beta = (\psi_{drone} + \theta_{yaw\_offset} + \Delta \alpha_x) \pmod{360^\circ}$

### C. 2D Horizontal Displacements
- $\Delta North = d_{ground} \cdot \cos(\beta)$
- $\Delta East = d_{ground} \cdot \sin(\beta)$
- Ground distance verification: $d_{ground} = \sqrt{\Delta North^2 + \Delta East^2}$
- Full 3D Euclidean distance: $D_{3D} = \sqrt{\Delta North^2 + \Delta East^2 + h^2} = d_{slant}$

### D. High-Accuracy Geodetic GPS Projection (WGS-84)
Using Great-Circle forward projection with mean Earth radius $R = 6,378,137\text{ m}$:
$$\delta = \frac{d_{ground}}{R}$$
$$\phi_{human} = \arcsin\left(\sin\phi_{drone} \cos\delta + \cos\phi_{drone} \sin\delta \cos\beta\right)$$
$$\lambda_{human} = \lambda_{drone} + \arctan2\left(\sin\beta \sin\delta \cos\phi_{drone}, \cos\delta - \sin\phi_{drone} \sin\phi_{human}\right)$$

---

## 2. Directory Structure

```
drone_flood_rescue_system/
├── config/
│   └── default_config.yaml         # Central YAML configuration (thresholds, FOV, tilt, GPS)
├── core/
│   ├── __init__.py
│   ├── detector.py                 # YOLOv8 human detection (conf >= 80% filter) + ByteTrack
│   ├── pythagoras_geolocator.py    # Pythagorean distance and WGS-84 GPS projection engine
│   ├── telemetry.py                # Telemetry adapters: Mock, MAVLink (Pixhawk), Serial GPS
│   └── rescue_logger.py            # Alerting, snapshot saving, CSV & GeoJSON export
├── models/
│   └── yolov8n.pt                  # Pre-trained YOLOv8 nano model weights
├── utils/
│   ├── __init__.py
│   ├── hud_visualizer.py           # Tactical HUD overlay (radar, bearings, distance, GPS)
│   └── video_stream.py             # Threaded reader for PiCam, USB webcams, and video files
├── tests/
│   ├── __init__.py
│   ├── test_pythagoras.py          # Unit tests verifying Pythagorean math & GPS projection
│   └── test_system_integration.py  # End-to-end integration test
├── test_videos/                    # Real drone disaster & flood test clips
├── run_drone_system.py             # Main live application entry point
├── simulate_flood_mission.py       # Mission simulation & evaluation script
├── export_for_pi.py                # Model exporter (PyTorch -> ONNX / NCNN / TFLite)
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 3. Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Running Unit Tests
Validate the Pythagorean formulas and geodetic projections:
```bash
python -m unittest discover tests
```

---

## 4. Usage Guide

### A. Live Drone / Raspberry Pi Deployment
Run with connected camera (USB or Pi Camera Module) and real-time Heads-Up Display:
```bash
python run_drone_system.py --source 0 --alt 25.0 --tilt 45.0 --conf 0.80
```

### B. Headless Flight Mode (For In-Flight Raspberry Pi)
When flying without a monitor attached to the Pi:
```bash
python run_drone_system.py --source 0 --no-display --record output_logs/mission_flight.mp4
```

### C. Flight Controller Integration (Pixhawk / ArduPilot / PX4)
Edit `config/default_config.yaml`:
```yaml
telemetry:
  provider: "mavlink"
  mavlink:
    connection_string: "/dev/ttyAMA0"   # Pi UART connected to Pixhawk TELEM2
    baud_rate: 57600
```
Then run:
```bash
python run_drone_system.py --source 0
```
The system will automatically read live GPS, altitude from LiDAR/barometer, and gimbal tilt angle from MAVLink.

### D. Simulating a Flood Mission on Test Videos
Run a simulated flight over a recorded flood disaster clip:
```bash
python simulate_flood_mission.py --video test_videos/people-detection.mp4 --conf 0.80 --alt 20.0 --tilt 45.0
```

---

## 5. Output Data & Rescue Exports

All verified human detections ($\ge 80\%$ confidence) are saved to `output_logs/`:
1. **`rescue_detections.csv`**:
   - `timestamp_iso`: UTC ISO 8601 timestamp
   - `track_id`: Unique tracked survivor ID
   - `confidence_pct`: Confidence score (e.g. 89.5%)
   - `target_latitude` & `target_longitude`: Exact computed GPS
   - `ground_distance_m`: Horizontal ground distance ($d_{ground}$)
   - `slant_distance_m`: Line-of-sight 3D distance via Pythagoras ($d_{slant}$)
   - `drone_latitude`, `drone_longitude`, `drone_altitude_m`, `camera_tilt_deg`
   - `snapshot_file`: Path to snapshot image
2. **`rescue_targets.geojson`**:
   - Ready to import directly into **QGIS**, **Google Earth**, **ATAK / CivTAK**, or web dashboards for disaster rescue teams.
3. **`snapshots/`**:
   - High-resolution cropped images with HUD telemetry stamps for visual verification.
