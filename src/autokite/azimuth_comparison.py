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
theodolite.index -= timedelta(hours=1)
theo_azimuth = theodolite["realaz"]
theo_azimuth = np.where(theo_azimuth<300,theo_azimuth+360,theo_azimuth)
theo_azimuth = pd.Series(theo_azimuth,index=theodolite.index)

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
    elevation, azimuth = pixel_to_sky_angles(coord[1], coord[0])
    semikite_all_elevation.append(int(round(elevation, 0)))
    semikite_all_azimuth.append(int(round(azimuth, 0)))
semikite["elevation"] = semikite_all_elevation
semikite["azimuth"] = semikite_all_azimuth

# semikite merging with autokite
autokite.update(semikite)
autokite.reset_index(inplace=True)
autokite.set_index('time', drop=True, inplace=True)
autokite.index = pd.to_datetime(autokite.index, format="%Y%m%d_%H%M%S")
autokite.index -= timedelta(hours=1)
autokite_azimuth = autokite["azimuth"]

# plotting part
def fix_azimuth_wrap(series):
    series_fixed = series.copy()
    if series.mean() > 180:  # cluster is on high side
        series_fixed[series < 90] += 360
    return series_fixed

theo_fixed = fix_azimuth_wrap(theo_azimuth)
auto_fixed = fix_azimuth_wrap(autokite_azimuth)

fig, ax = plt.subplots(figsize=(8,5), dpi=150)
ax.plot(theo_fixed.index, theo_fixed-180, label="Theodolite")
ax.plot(auto_fixed.index, auto_fixed-180, label="Autokite + Semikite")
ax.set_xlim(list(autokite.index)[0], list(autokite.index)[-1])
ax.set_xticks(list(autokite.index)[0::100])
ax.set_xlabel('09/01/2025 Time (UTC)', fontsize=10, weight="bold")
ax.set_ylabel('azimuth angle derived wind direction (°)', fontsize=10, weight="bold")
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.yaxis.grid(True)
plt.legend()
plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95)
plt.tight_layout()
# plt.title(f"azimuth angle comparison for the launch starting at {DATE} {time}")
plt.savefig(f"azimuth_angle_comparison_{DATE}_{time_measured}.png", dpi=150)
plt.show()

# statistical data calculation part
# reset time index to avoid any issues due to slight time offset
theo_azimuth = theo_azimuth.reset_index(drop=True)
autokite_azimuth = autokite_azimuth.reset_index(drop=True)

mean_diff = np.sqrt((autokite_azimuth-theo_azimuth)**2)
mean_diff = mean_diff.mean()
print(f"The mean angle difference between auto-/semikite and the theodolite measurement is: {round(mean_diff,2)}")
print(f"The correlation between auto-/semikite and theodolite measurement is: {round(autokite_azimuth.corr(theo_azimuth),2)}")