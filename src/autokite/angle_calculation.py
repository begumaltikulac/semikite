"""
    Author: Bohong Li

    Given the coordinates found in auto_kite_detection.py, this script should calculate the angle of the kite
    relative to the fisheye camera.
"""

import matplotlib.pyplot as plt
import pickle

from functions_autokite import (
    open_theodolite,
    pixel_to_sky_angles,
    # pixel_to_angles_with_height
)
DATE = "20250829"
coords_outfile = f"coordinates/coordinates_{DATE}.pckl"
outcsv_name = f"coordinates/coordinates_with_angles_{DATE}.csv"
theo_file = "TheoGelb_20250829_124013.txt"

SUBFOLDER = "theo_with_no_radiosonde_first_flight"  # "theo_with_no_radiosonde_first_flight" "theo_with_radiosonde_second_flight"
# coords_outfile = f"coordinates/coordinates_{DATE}_{SUBFOLDER}.pckl"
# outcsv_name = f"coordinates/coordinates_with_angles_{DATE}_{SUBFOLDER}.csv"

all_elevation = []
all_azimuth = []
heights = np.arange(40,100) # NOTE: The height information is still missing. We need that information from the radiosondes

with open(coords_outfile, "rb") as f:
    coordinates = pickle.load(f)

# for coord, height in zip(coordinates["coordinates [x,y]"], heights):
for coord in coordinates["coordinates [x,y]"]:
    elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1])
    all_elevation.append(int(round(elevation,0)))
    all_azimuth.append(int(round(azimuth,0)))

coordinates["elevation"] = all_elevation
coordinates["azimuth"] = all_azimuth
# coordinates.to_csv(outcsv_name)

theo = open_theodolite(file="theodolite_data/TheoGelb_20250829_124013.txt", obs_date="2025-08-29", start_time="10:31:08")
theo_elevation = theo["elevation"]
autokite_elevation = coordinates["elevation"]

plt.figure(dpi=150)
plt.plot(theo_elevation, label="yellow theodolite")
plt.plot(autokite_elevation, label="autokite")
plt.legend()
plt.show()