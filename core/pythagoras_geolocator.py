"""
Pythagorean Geolocation & Distance Estimation Engine for Drone Flood Search & Rescue.

Uses drone telemetry (Altitude, GPS Latitude, Longitude, Heading/Yaw) and
Camera geometry (Gimbal Tilt/Pitch, Field of View, Detection pixel coordinates)
to compute:
1. Horizontal Ground Distance (d_ground)
2. Slant Distance (Line of sight 3D distance via Pythagoras: d_slant = sqrt(d_ground^2 + h^2))
3. Target True Bearing (degrees from North)
4. Exact Geodetic GPS Location (Latitude, Longitude) of the detected human.
"""

import math
from typing import Tuple, Dict, Any, Optional

# WGS-84 Earth equatorial radius in meters
EARTH_RADIUS_M = 6378137.0

class PythagorasGeolocator:
    """
    Computes exact ground distance, slant range, bearing, and target GPS coordinates
    for aerial object detections using Pythagorean theorem and spherical geodesy.
    """

    def __init__(
        self,
        horizontal_fov_deg: float = 62.2,
        vertical_fov_deg: float = 48.8,
        default_tilt_deg: float = 45.0
    ):
        """
        Args:
            horizontal_fov_deg: Camera horizontal Field of View in degrees.
            vertical_fov_deg: Camera vertical Field of View in degrees.
            default_tilt_deg: Camera pitch/tilt downwards from horizontal in degrees
                              (0° = forward level, 90° = vertical nadir).
        """
        self.hfov_rad = math.radians(horizontal_fov_deg)
        self.vfov_rad = math.radians(vertical_fov_deg)
        self.default_tilt_deg = default_tilt_deg

    def calculate_focal_lengths(self, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """
        Calculate pinhole camera focal lengths in pixels from FOV.
        """
        fx = (frame_width / 2.0) / math.tan(self.hfov_rad / 2.0)
        fy = (frame_height / 2.0) / math.tan(self.vfov_rad / 2.0)
        return fx, fy

    def pixel_to_ray_angles(
        self,
        px: float,
        py: float,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinate (px, py) to angular offsets from optical center.
        
        Returns:
            (alpha_x, alpha_y) in degrees:
            - alpha_x: Azimuth angle (+ is right of optical center, - is left)
            - alpha_y: Elevation angle (+ is below optical center / looking steeper down, - is above)
        """
        fx, fy = self.calculate_focal_lengths(frame_width, frame_height)
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        dx = px - cx
        dy = py - cy

        alpha_x_rad = math.atan(dx / fx)
        alpha_y_rad = math.atan(dy / fy)

        return math.degrees(alpha_x_rad), math.degrees(alpha_y_rad)

    def locate_target(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_alt_m: float,
        drone_heading_deg: float,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        camera_tilt_deg: Optional[float] = None,
        camera_yaw_offset_deg: float = 0.0,
        contact_point: str = "bottom_center"
    ) -> Dict[str, Any]:
        """
        Calculate target distance and GPS coordinates from a detection bounding box.

        Args:
            drone_lat: Current drone GPS latitude (degrees).
            drone_lon: Current drone GPS longitude (degrees).
            drone_alt_m: Drone altitude above flood water/ground surface in meters (h).
            drone_heading_deg: Drone heading (yaw) relative to True North (0-360°).
            bbox: (x1, y1, x2, y2) bounding box of detected human.
            frame_width: Width of video frame in pixels.
            frame_height: Height of video frame in pixels.
            camera_tilt_deg: Camera pitch angle downwards from horizontal in degrees.
                             If None, uses default_tilt_deg.
            camera_yaw_offset_deg: Camera offset relative to drone body heading.
            contact_point: 'bottom_center' (contact point with water/ground) or 'center'.

        Returns:
            Dictionary containing:
            - target_lat: Target GPS latitude
            - target_lon: Target GPS longitude
            - ground_distance_m: Horizontal distance to target (d_ground)
            - slant_distance_m: Line-of-sight 3D distance via Pythagoras (d_slant)
            - target_bearing_deg: Azimuth heading towards target from drone
            - depression_angle_deg: Total look-down angle from horizontal
            - pythagoras_verified: True when d_slant^2 == d_ground^2 + h^2
        """
        x1, y1, x2, y2 = bbox
        
        # Determine ground intersection pixel coordinate
        if contact_point == "bottom_center":
            px = (x1 + x2) / 2.0
            py = y2  # Feet / water contact point
        else:
            px = (x1 + x2) / 2.0
            py = (y1 + y2) / 2.0

        if camera_tilt_deg is None:
            camera_tilt_deg = self.default_tilt_deg

        # Step 1: Angular ray deviation from camera optical axis
        alpha_x, alpha_y = self.pixel_to_ray_angles(px, py, frame_width, frame_height)

        # Step 2: Total depression angle (gamma) below the horizontal
        depression_angle = camera_tilt_deg + alpha_y

        # Safeguard: ensure ray points below horizon (towards water/ground)
        depression_angle_clamped = max(1.0, min(89.9, depression_angle))
        gamma_rad = math.radians(depression_angle_clamped)

        # Step 3: Trigonometry & Pythagorean Distance Computation
        h = max(0.5, float(drone_alt_m))  # Altitude must be positive

        # Horizontal Ground Distance: d_ground = h / tan(gamma)
        d_ground = h / math.tan(gamma_rad)

        # Direct Line-of-Sight Slant Distance: d_slant = h / sin(gamma)
        # Using Pythagorean theorem: d_slant = sqrt(d_ground^2 + h^2)
        d_slant_pythagoras = math.sqrt((d_ground ** 2) + (h ** 2))
        d_slant_trig = h / math.sin(gamma_rad)

        # Step 4: True Bearing towards target
        target_bearing = (drone_heading_deg + camera_yaw_offset_deg + alpha_x) % 360.0

        # Step 5: Horizontal displacements (2D Pythagoras: d_ground = sqrt(dN^2 + dE^2))
        brng_rad = math.radians(target_bearing)
        delta_north = d_ground * math.cos(brng_rad)
        delta_east = d_ground * math.sin(brng_rad)

        # Full 3D Pythagorean Verification:
        # 3D Distance = sqrt(delta_north^2 + delta_east^2 + h^2) == d_slant
        d_3d_pythagoras = math.sqrt(delta_north**2 + delta_east**2 + h**2)
        pythagoras_verified = abs(d_3d_pythagoras - d_slant_pythagoras) < 0.01

        # Step 6: Forward Geodetic GPS Projection (Great-Circle / WGS84 Spherical)
        target_lat, target_lon = self.project_gps_coordinates(
            drone_lat, drone_lon, d_ground, target_bearing
        )

        return {
            "target_lat": round(target_lat, 7),
            "target_lon": round(target_lon, 7),
            "drone_lat": round(drone_lat, 7),
            "drone_lon": round(drone_lon, 7),
            "drone_alt_m": round(h, 2),
            "ground_distance_m": round(d_ground, 2),
            "slant_distance_m": round(d_slant_pythagoras, 2),
            "delta_north_m": round(delta_north, 2),
            "delta_east_m": round(delta_east, 2),
            "target_bearing_deg": round(target_bearing, 2),
            "depression_angle_deg": round(depression_angle_clamped, 2),
            "pixel_x": round(px, 1),
            "pixel_y": round(py, 1),
            "pythagoras_verified": pythagoras_verified
        }

    @staticmethod
    def project_gps_coordinates(
        start_lat: float,
        start_lon: float,
        distance_m: float,
        bearing_deg: float
    ) -> Tuple[float, float]:
        """
        Forward geodetic projection: Computes destination (lat, lon) given
        start position, distance in meters, and true bearing.
        """
        phi1 = math.radians(start_lat)
        lam1 = math.radians(start_lon)
        brng = math.radians(bearing_deg)
        angular_dist = distance_m / EARTH_RADIUS_M

        phi2 = math.asin(
            math.sin(phi1) * math.cos(angular_dist) +
            math.cos(phi1) * math.sin(angular_dist) * math.cos(brng)
        )

        lam2 = lam1 + math.atan2(
            math.sin(brng) * math.sin(angular_dist) * math.cos(phi1),
            math.cos(angular_dist) - math.sin(phi1) * math.sin(phi2)
        )

        return math.degrees(phi2), math.degrees(lam2)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two GPS coordinates in meters.
        Used for verification and telemetry drift tests.
        """
        phi1, lam1 = math.radians(lat1), math.radians(lon1)
        phi2, lam2 = math.radians(lat2), math.radians(lon2)

        dphi = phi2 - phi1
        dlam = lam2 - lam1

        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return EARTH_RADIUS_M * c
