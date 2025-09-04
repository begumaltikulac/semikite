"""
    Author: Bohong Li

    Given the coordinates found in auto_kite_detection.py, this script should calculate the angle of the kite
    relative to the fisheye camera.
"""

import pickle

from functions_autokite import (
    open_theodolite,
    pixel_to_sky_angles,
)

DATE = "20250903"
theo_start_date = "2025-09-03"

for SUBFOLDER in ["morning","afternoon"]:
    all_elevation = []
    all_azimuth = []
    coords_outfile = f"coordinates/{DATE}/coordinates_{DATE}_{SUBFOLDER}.pckl"
    outcsv_name = f"coordinates/{DATE}/coordinates_with_angles_{DATE}_{SUBFOLDER}.csv"
    with open(coords_outfile, "rb") as f:
        coordinates = pickle.load(f)
    for coord in coordinates["coordinates [x,y]"]:
        elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1])
        all_elevation.append(int(round(elevation,0)))
        all_azimuth.append(int(round(azimuth,0)))
    coordinates["elevation"] = all_elevation
    coordinates["azimuth"] = all_azimuth
    coordinates.to_csv(outcsv_name)

for theo_start_time, theo_file, color in zip(
    # ["10:03:20", "10:03:20"],
    # ["theodolite_data/TheoGelb_20250901_100320.td4", "theodolite_data/TheoRot_20250901_100320.td4"],
    ["13:17:08", "13:17:09"],
    ["theodolite_data/TheoGelb_20250901_131708.txt", "theodolite_data/TheoRot_20250901_131709.txt"],
    ["yellow", "red"],
):

    theo = open_theodolite(file=theo_file, obs_date=theo_start_date, start_time=theo_start_time)
    theo.to_csv(f"coordinates/{DATE}/{color}_theodolite_angles_{DATE}_{SUBFOLDER}.csv")

all_elevation = []
all_azimuth = []

with open(coords_outfile, "rb") as f:
    coordinates = pickle.load(f)
for coord in coordinates["coordinates [x,y]"]:
    elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1])
    all_elevation.append(int(round(elevation,0)))
    all_azimuth.append(int(round(azimuth,0)))
coordinates["elevation"] = all_elevation
coordinates["azimuth"] = all_azimuth
# coordinates.to_csv(outcsv_name)

# plt.figure(dpi=150)
# plt.plot(theo_elevation, label="yellow theodolite")
# plt.plot(autokite_elevation, label="autokite")
# plt.legend()
# plt.show()