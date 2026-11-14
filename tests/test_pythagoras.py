"""
Unit tests verifying Pythagorean distance formulas, slant range identity,
and GPS projection accuracy.
"""

import math
import unittest
from core.pythagoras_geolocator import PythagorasGeolocator

class TestPythagorasGeolocator(unittest.TestCase):

    def setUp(self):
        self.geolocator = PythagorasGeolocator(
            horizontal_fov_deg=60.0,
            vertical_fov_deg=45.0,
            default_tilt_deg=45.0
        )

    def test_center_pixel_pythagoras_45_deg(self):
        """
        At 45° tilt, center pixel (depression angle = 45°),
        Ground distance d_ground must equal altitude h:
        d_ground = h / tan(45°) = h
        d_slant = sqrt(h^2 + h^2) = h * sqrt(2)
        """
        alt_m = 20.0
        drone_lat = 12.971599
        drone_lon = 77.594566
        drone_heading = 0.0  # Due North

        # Center of 640x480 frame
        bbox = (310, 230, 330, 250)  # center ~ (320, 240)
        
        result = self.geolocator.locate_target(
            drone_lat=drone_lat,
            drone_lon=drone_lon,
            drone_alt_m=alt_m,
            drone_heading_deg=drone_heading,
            bbox=bbox,
            frame_width=640,
            frame_height=480,
            camera_tilt_deg=45.0,
            contact_point="center"
        )

        d_ground = result["ground_distance_m"]
        d_slant = result["slant_distance_m"]

        self.assertAlmostEqual(d_ground, 20.0, delta=0.5)
        self.assertAlmostEqual(d_slant, 20.0 * math.sqrt(2), delta=0.5)
        self.assertTrue(result["pythagoras_verified"])

        # Target should be directly North of drone
        self.assertAlmostEqual(result["target_bearing_deg"], 0.0, delta=0.5)
        self.assertGreater(result["target_lat"], drone_lat)
        self.assertAlmostEqual(result["target_lon"], drone_lon, delta=0.0001)

        # Haversine distance verification
        measured_dist = PythagorasGeolocator.haversine_distance(
            drone_lat, drone_lon, result["target_lat"], result["target_lon"]
        )
        self.assertAlmostEqual(measured_dist, d_ground, delta=0.5)

    def test_nadir_pointing_90_deg(self):
        """
        At 90° tilt (pointing straight down),
        d_ground should be ~ 0, and d_slant should be ~ altitude h.
        """
        alt_m = 35.0
        bbox = (310, 230, 330, 250)
        
        result = self.geolocator.locate_target(
            drone_lat=12.971599,
            drone_lon=77.594566,
            drone_alt_m=alt_m,
            drone_heading_deg=90.0,
            bbox=bbox,
            frame_width=640,
            frame_height=480,
            camera_tilt_deg=89.0, # nearly nadir
            contact_point="center"
        )

        self.assertLess(result["ground_distance_m"], 2.0)
        self.assertAlmostEqual(result["slant_distance_m"], alt_m, delta=1.0)
        self.assertTrue(result["pythagoras_verified"])

    def test_pythagoras_identity_arbitrary_angles(self):
        """
        Validate d_slant^2 == d_ground^2 + h^2 for a wide range of altitudes and angles.
        """
        for alt in [5.0, 15.0, 30.0, 50.0, 100.0]:
            for tilt in [20.0, 35.0, 45.0, 60.0, 75.0]:
                for px in [100, 320, 540]:
                    for py in [80, 240, 400]:
                        bbox = (px - 10, py - 10, px + 10, py + 10)
                        res = self.geolocator.locate_target(
                            drone_lat=13.0,
                            drone_lon=77.0,
                            drone_alt_m=alt,
                            drone_heading_deg=135.0,
                            bbox=bbox,
                            frame_width=640,
                            frame_height=480,
                            camera_tilt_deg=tilt,
                            contact_point="center"
                        )
                        d_g = res["ground_distance_m"]
                        d_s = res["slant_distance_m"]
                        expected_slant = math.sqrt(d_g**2 + alt**2)
                        self.assertAlmostEqual(d_s, expected_slant, delta=0.05)
                        self.assertTrue(res["pythagoras_verified"])

if __name__ == "__main__":
    unittest.main()
