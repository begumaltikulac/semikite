import matplotlib.pyplot as plt
import pandas as pd

# matplotlib.use("Qt5Agg")  # "TkAgg"

DATE = "20250903"
SUBFOLDER = "afternoon"
theodolite = pd.read_csv(f'coordinates/{DATE}/theodolite_angles_20250903_{SUBFOLDER}.csv')[::2].reset_index(drop=True)
theo_elevation = theodolite["elevation"]
autokite = pd.read_csv(f'coordinates/{DATE}/coordinates_with_angles_20250903_{SUBFOLDER}.csv')
autokite_elevation = autokite["elevation"]

plt.figure(dpi=150)
plt.plot(theo_elevation, label="yellow theodolite")
plt.plot(autokite_elevation, label="autokite")
plt.legend()
plt.title(f"elevation comparison of launch {DATE} {SUBFOLDER}")
plt.show()