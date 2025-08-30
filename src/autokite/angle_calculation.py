"""
    Author: Bohong Li

    Given the coordinates found in auto_kite_detection.py, this script should calculate the angle of the kite
    relative to the fisheye camera.
"""

import numpy as np
import pickle

from functions_autokite import pixel_to_angles_with_height

with open("coordinates.pckl", "rb") as f:
    coordinates = pickle.load(f)

all_elevation = []
all_azimuth = []
heights = np.arange(40,100) # NOTE: The height information is still missing. We need that information from the radiosondes

for coord, height in zip(coordinates["coordinates [x,y]"], heights):
    elevation, azimuth, _ = pixel_to_angles_with_height(coord[0], coord[1], height)
    all_elevation.append(int(round(elevation,0)))
    all_azimuth.append(int(round(azimuth,0)))

coordinates["elevation"] = all_elevation
coordinates["azimuth"] = all_azimuth

coordinates.to_csv("coordinates_with_angles_with_height.csv")