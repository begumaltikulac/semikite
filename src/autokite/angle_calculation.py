"""
    Author: Bohong Li

    Given the coordinates found in auto_kite_detection.py, this script should calculate the angle of the kite
    relative to the fisheye camera.
"""

import pickle

from functions_autokite import pixel_to_sky_angles

with open("coordinates.pckl", "rb") as f:
    coordinates = pickle.load(f)

all_elevation = []
all_azimuth = []

for coord in coordinates["coordinates"]:
    elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1], 960, 960, 960)
    all_elevation.append(int(round(elevation,0)))
    all_azimuth.append(int(round(azimuth,0)))

coordinates["elevation"] = all_elevation
coordinates["azimuth"] = all_azimuth

coordinates.to_csv("coordinates_with_angles.csv")