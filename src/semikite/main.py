import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

#sys.path.append("../autokite") # add the autokite directory to this file, to use the other functions

# !!! still need to correctly import the functions
from functions_autokite import document_top_pixels_as_pickle, find_timestamps
#from ..autokite.functions_autokite import document_top_pixels_as_pickle, find_timestamps
#from ../autokite/functions_autokite import document_top_pixels_as_pickle, find_timestamps

matplotlib.use('TkAgg')  # <- important command for GUI window

file_names = os.listdir("images")
#files_names_path = ['images/' + f for f in file_names]
image_paths = ['images/FE4_Image_20160901_114200_UTCp1.jpg']

clicked_coord = []

timestamps = find_timestamps(file_names)
print(timestamps)
def onclick(event):
    global clicked_coord
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        clicked_coord.append((x,y))
        print(f"Coordinates: x={x}, y={y}")
        plt.close()

for path in image_paths:
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(15,25))
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Please click on the kite in the image")
    plt.show()

if clicked_coord:
    print("Saved coordinates:", clicked_coord)
else:
    print("It did not work.")

print(file_names)

# use this function:
#document_top_pixels_as_pickle