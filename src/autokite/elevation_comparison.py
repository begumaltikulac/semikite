from datetime import timedelta
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from functions_autokite import pixel_to_sky_angles

matplotlib.use("Qt5Agg")  # "TkAgg"

DATE = "20250901"
time = "10:04:20 CEST"
time_measured = "morning"
semikite_filename = f"../semikite/coordinates_semikite_{DATE}_{time_measured}.pckl"

# theodolite part
theodolite = pd.read_csv(f'coordinates/{DATE}/yellow_theodolite_angles_{DATE}_{time_measured}.csv').reset_index(drop=True)[2:-1:2]
theodolite.set_index('time_sec', drop=True, inplace=True)
theodolite.index = pd.to_datetime(theodolite.index, format="%Y-%m-%d %H:%M:%S")
theodolite.index += timedelta(minutes=1)
theo_elevation = theodolite["elevation"]

# autokite part
autokite = pd.read_csv(f'coordinates/{DATE}/coordinates_with_angles_{DATE}_{time_measured}.csv')
autokite.set_index('time', drop=True, inplace=True)
# semikite part
with open(semikite_filename, "rb") as f:
    semikite = pickle.load(f)
semikite_all_elevation = []
semikite_all_azimuth = []
semikite_coordinates = semikite["coordinates [x,y]"]
for coord in semikite_coordinates:
    elevation, azimuth = pixel_to_sky_angles(coord[0], coord[1])
    semikite_all_elevation.append(int(round(elevation, 0)))
    semikite_all_azimuth.append(int(round(azimuth, 0)))
semikite["elevation"] = semikite_all_elevation
semikite["azimuth"] = semikite_all_azimuth

# semikite merging with autokite
autokite.update(semikite)
autokite.reset_index(inplace=True)
autokite.set_index('time', drop=True, inplace=True)
autokite.index = pd.to_datetime(autokite.index, format="%Y%m%d_%H%M%S")
# autokite.index += timedelta(hours=1)
autokite_elevation = autokite["elevation"]

# GPS
gps = pd.read_csv(f'gps_results/{DATE}_{time_measured}.csv')[2:]
gps.set_index('time', drop=True, inplace=True)
gps.index = pd.to_datetime(gps.index)
gps.index += timedelta(hours=1)
gps_elevation = gps["elevation_angle"]

# plotting part
fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
ax.plot(theo_elevation, label="yellow theodolite")
ax.plot(autokite_elevation, label="autokite")
ax.plot(gps_elevation, label="gps")
ax.set_xlim(list(autokite.index)[0], list(autokite.index)[-1])
ax.set_xticks(list(autokite.index)[0::100])
ax.set_xlabel('Time (UTC)', fontsize=10, weight="bold")
ax.set_ylabel('Elevation angle (°)', fontsize=10, weight="bold")
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.yaxis.grid(True)
plt.legend()
# plt.title(f"elevation angle comparison for the launch starting at {DATE} {time}")
plt.savefig(f"elevation_angle_comparison_{DATE}_{time_measured}.png", dpi=150)
plt.show()

# statistical data calculation part
# reset time index to avoid any issues due to slight time offset
theo_elevation = theo_elevation.reset_index(drop=True)
autokite_elevation = autokite_elevation.reset_index(drop=True)
gps_elevation = gps_elevation.reset_index(drop=True)

for device, angle in zip(["auto-/semikite", "gps"],[autokite_elevation, gps_elevation]):
    mean_diff = abs(angle.mean() - theo_elevation.mean())
    rmse = np.sqrt(((angle-autokite_elevation)**2).mean())
    print(f"The mean angle difference between {device} and the theodolite measurement is: {round(mean_diff,2)}")
    print(f"The rmse between {device} and theodolite is {round(rmse,2)}.")
    print(f"The correlation between {device} and theodolite measurement is: {round(angle.corr(theo_elevation),2)}")
    print()