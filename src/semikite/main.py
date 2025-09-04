import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import sys
import os

from src.autokite.functions_autokite import document_top_pixels_as_pickle, find_timestamps

# CHANGE ACOORDINGLY TO WHICH LAUNCH ON WHICH DAY
DATE = "20250901"
SUBFOLDER = "morning"

matplotlib.use('TkAgg')  # <- important command for GUI window
REL_PATH_AUTOKITE = "../autokite"
# add the autokite directory to this file, to use the other functions
sys.path.append(REL_PATH_AUTOKITE)

original_images_path = f"{REL_PATH_AUTOKITE}/images_{DATE}/{SUBFOLDER}"
csv_wrong_detection = pd.read_csv(f"{REL_PATH_AUTOKITE}/coordinates/{DATE}/false_detection_{DATE}_{SUBFOLDER}.csv")
time_wrong_detection = csv_wrong_detection["time"]
file_names = [f"{original_images_path}/Image_{time}_UTCp1.jpg" for time in time_wrong_detection]

clicked_coord = []
timestamps = find_timestamps(file_names)
coords_semikite = dict((time,0) for time in timestamps)

def onclick(event):
    global clicked_coord
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        clicked_coord.append([y,x])
        #print(f"Coordinates: x={x}, y={y}")
        plt.close()

for index, path in enumerate(file_names):
    img = mpimg.imread(path)
    time_loop = timestamps[index]
    fig, ax = plt.subplots(figsize=(15,25))
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Please click on the kite in the image")
    plt.show()
    if clicked_coord:  # store last clicked coordinates
        coords_semikite[time_loop] = clicked_coord[-1]

# save the coordinates as pickle file
document_top_pixels_as_pickle(coords_semikite, output_file=f"coordinates_semikite_{DATE}_{SUBFOLDER}.pckl")

if clicked_coord:
    print("Saved coordinates:", clicked_coord)
else:
    print("It did not work.")