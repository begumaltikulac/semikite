import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from functions_autokite import pixel_to_sky_angles

matplotlib.use("Qt5Agg")  # "TkAgg"

DATE = "20250901"
SUBFOLDER = "morning"
semikite_filename = f"coordinates/{DATE}/coordinates_semikite_20250901_morning.pckl"

# theodolite part
theodolite = pd.read_csv(f'coordinates/{DATE}/yellow_theodolite_angles_{DATE}_{SUBFOLDER}.csv')[::2].reset_index(drop=True)
theodolite.set_index('time_sec', drop=True, inplace=True)
theodolite.index = pd.to_datetime(theodolite.index, format="%Y-%m-%d %H:%M:%S")
theo_elevation = theodolite["elevation"]

# autokite part
autokite = pd.read_csv(f'coordinates/{DATE}/coordinates_with_angles_{DATE}_{SUBFOLDER}.csv')
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
autokite_elevation = autokite["elevation"]

# plotting part
plt.figure(dpi=150)
plt.plot(theo_elevation, label="yellow theodolite")
plt.plot(autokite_elevation, label="autokite")
plt.legend()
plt.title(f"elevation comparison of launch {DATE} {SUBFOLDER}")
plt.show()

# statistical data calculation part
mean_diff = abs(np.mean(theo_elevation) - np.mean(autokite_elevation))
rmse = np.sqrt(((theo_elevation-autokite_elevation)**2).mean())
print(f"The mean angle difference is: {mean_diff}")
print(f"The rmse is {rmse}.")
print(f"The correlation between two measurement is: {theo_elevation.corr(autokite_elevation)}")