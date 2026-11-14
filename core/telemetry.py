"""
Drone Telemetry Management for Raspberry Pi & Flight Controller Integration.

Supports:
1. Mock Telemetry (for testing, desktop simulation, and bench runs)
2. MAVLink Telemetry (ArduPilot / Pixhawk / PX4 via serial UART or UDP)
3. Serial NMEA GPS (standalone GPS modules on Raspberry Pi GPIO/USB)
"""

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class DroneState:
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    camera_tilt_deg: float = 45.0
    timestamp: float = 0.0
    is_valid: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.latitude,
            "lon": self.longitude,
            "alt_m": self.altitude_m,
            "heading_deg": self.heading_deg,
            "camera_tilt_deg": self.camera_tilt_deg,
            "timestamp": self.timestamp or time.time()
        }

class BaseTelemetryProvider:
    def get_state(self) -> DroneState:
        raise NotImplementedError
    
    def close(self):
        pass

class MockTelemetryProvider(BaseTelemetryProvider):
    """
    Mock telemetry for laboratory testing, simulation, and bench demonstration.
    """
    def __init__(
        self,
        latitude: float = 12.971599,
        longitude: float = 77.594566,
        altitude_m: float = 25.0,
        heading_deg: float = 45.0,
        camera_tilt_deg: float = 45.0,
        simulate_drift: bool = False
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_m = altitude_m
        self.heading_deg = heading_deg
        self.camera_tilt_deg = camera_tilt_deg
        self.simulate_drift = simulate_drift
        self.start_time = time.time()

    def set_telemetry(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude_m: Optional[float] = None,
        heading_deg: Optional[float] = None,
        camera_tilt_deg: Optional[float] = None
    ):
        if latitude is not None: self.latitude = latitude
        if longitude is not None: self.longitude = longitude
        if altitude_m is not None: self.altitude_m = altitude_m
        if heading_deg is not None: self.heading_deg = heading_deg % 360.0
        if camera_tilt_deg is not None: self.camera_tilt_deg = camera_tilt_deg

    def get_state(self) -> DroneState:
        now = time.time()
        lat = self.latitude
        lon = self.longitude
        hdg = self.heading_deg
        alt = self.altitude_m

        if self.simulate_drift:
            # Simulate slight drone loitering movement
            dt = now - self.start_time
            lat += 0.00002 * math.sin(dt * 0.1)
            lon += 0.00002 * math.cos(dt * 0.1)
            hdg = (self.heading_deg + 5.0 * math.sin(dt * 0.2)) % 360.0

        return DroneState(
            latitude=lat,
            longitude=lon,
            altitude_m=alt,
            heading_deg=hdg,
            camera_tilt_deg=self.camera_tilt_deg,
            timestamp=now,
            is_valid=True
        )

class MavlinkTelemetryProvider(BaseTelemetryProvider):
    """
    MAVLink connection for Pixhawk / ArduPilot / PX4 connected to Raspberry Pi.
    Requires pymavlink.
    """
    def __init__(self, connection_str: str = "udp:127.0.0.1:14550", baud: int = 57600):
        self.connection_str = connection_str
        self.baud = baud
        self.master = None
        self.latest_state = DroneState(
            latitude=12.971599,
            longitude=77.594566,
            altitude_m=20.0,
            heading_deg=0.0,
            is_valid=False
        )
        self._init_connection()

    def _init_connection(self):
        try:
            from pymavlink import mavutil
            logger.info(f"Connecting to MAVLink flight controller at {self.connection_str}...")
            self.master = mavutil.mavlink_connection(self.connection_str, baud=self.baud)
            logger.info("MAVLink connection established.")
        except ImportError:
            logger.warning("pymavlink not installed. Install with 'pip install pymavlink'. Using fallback state.")
        except Exception as e:
            logger.error(f"Failed to connect MAVLink: {e}")

    def get_state(self) -> DroneState:
        if self.master is None:
            return self.latest_state

        try:
            # Non-blocking poll for incoming MAVLink messages
            msg = self.master.recv_match(
                type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'MOUNT_STATUS'],
                blocking=False
            )
            while msg is not None:
                msg_type = msg.get_type()
                if msg_type == 'GLOBAL_POSITION_INT':
                    # lat and lon are scaled by 1e7
                    self.latest_state.latitude = msg.lat / 1e7
                    self.latest_state.longitude = msg.lon / 1e7
                    # relative_alt is in millimeters
                    self.latest_state.altitude_m = max(0.5, msg.relative_alt / 1000.0)
                    # hdg in cdeg (0.01 deg)
                    if msg.hdg != 65535:
                        self.latest_state.heading_deg = msg.hdg / 100.0
                    self.latest_state.is_valid = True
                    self.latest_state.timestamp = time.time()
                elif msg_type == 'ATTITUDE':
                    self.latest_state.pitch_deg = math.degrees(msg.pitch)
                    self.latest_state.roll_deg = math.degrees(msg.roll)
                    if not self.latest_state.is_valid:
                        self.latest_state.heading_deg = math.degrees(msg.yaw) % 360.0
                elif msg_type == 'MOUNT_STATUS':
                    # Camera gimbal pitch
                    self.latest_state.camera_tilt_deg = abs(msg.pointing_a / 100.0)

                msg = self.master.recv_match(
                    type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'MOUNT_STATUS'],
                    blocking=False
                )
        except Exception as e:
            logger.error(f"Error polling MAVLink telemetry: {e}")

        return self.latest_state

    def close(self):
        if self.master:
            try:
                self.master.close()
            except Exception:
                pass

def get_telemetry_provider(config: Dict[str, Any]) -> BaseTelemetryProvider:
    """
    Factory function to instantiate telemetry provider based on configuration.
    """
    t_cfg = config.get("telemetry", {})
    provider_type = t_cfg.get("provider", "mock").lower()

    if provider_type == "mavlink":
        m_cfg = t_cfg.get("mavlink", {})
        return MavlinkTelemetryProvider(
            connection_str=m_cfg.get("connection_string", "udp:127.0.0.1:14550"),
            baud=m_cfg.get("baud_rate", 57600)
        )
    else:
        d_cfg = config.get("drone", {})
        c_cfg = config.get("camera", {})
        return MockTelemetryProvider(
            latitude=d_cfg.get("default_latitude", 12.971599),
            longitude=d_cfg.get("default_longitude", 77.594566),
            altitude_m=d_cfg.get("default_altitude_m", 25.0),
            heading_deg=d_cfg.get("default_heading_deg", 45.0),
            camera_tilt_deg=c_cfg.get("tilt_pitch_deg", 45.0)
        )
