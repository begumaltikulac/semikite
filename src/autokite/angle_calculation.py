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

DATE = "20250901"
theo_start_date = "2025-09-01"
colors = ["yellow", "red"]

for time_measured in ["morning","afternoon"]:
    all_elevation = []
    all_azimuth = []
    coords_outfile = f"coordinates/{DATE}/coordinates_{DATE}_{time_measured}.pckl"
    outcsv_name = f"coordinates/{DATE}/coordinates_with_angles_{DATE}_{time_measured}.csv"
    with open(coords_outfile, "rb") as f:
        coordinates = pickle.load(f)
    for coord in coordinates["coordinates [x,y]"]:
        elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1])
        all_elevation.append(int(round(elevation,0)))
        all_azimuth.append(int(round(azimuth,0)))
    coordinates["elevation"] = all_elevation
    coordinates["azimuth"] = all_azimuth
    coordinates.to_csv(outcsv_name)

""" 20250901 exclusive
"""
if DATE == "20250901" and time_measured == "morning":
    theo_start_times = ["10:03:20", "10:03:20"]
    theo_files = [
        "theodolite_data/TheoGelb_20250901_100320.td4",
        "theodolite_data/TheoRot_20250901_100320.td4",
    ]
elif DATE == "20250901" and time_measured == "afternoon":
    theo_start_times = ["13:17:08", "13:17:09"]
    theo_files = [
        "theodolite_data/TheoGelb_20250901_131708.txt",
        "theodolite_data/TheoRot_20250901_131709.txt",
    ]
else:
    raise ValueError(f"Unknown time_measured: {time_measured}")

for theo_start_time, theo_file, color in zip(theo_start_times, theo_files, colors):
    theo = open_theodolite(file=theo_file, obs_date=theo_start_date, start_time=theo_start_time)
    theo.to_csv(f"coordinates/{DATE}/{color}_theodolite_angles_{DATE}_{time_measured}.csv")

# for theo_start_time, theo_file, time_measured in zip(
#     ["10:49:09", "14:43:17"],
#     [f"theodolite_data/TheoGelb_{DATE}_104909.txt", f"theodolite_data/TheoGElb_{DATE}_144217.txt"],
#     ["morning", "afternoon"],
# ):
#     theo = open_theodolite(file=theo_file, obs_date=theo_start_date, start_time=theo_start_time)
#     theo.to_csv(f"coordinates/{DATE}/theodolite_angles_{DATE}_{time_measured}.csv")