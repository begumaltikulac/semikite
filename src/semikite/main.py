import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

matplotlib.use('TkAgg')  # <- important command for GUI window
sys.path.append("../autokite") # add the autokite directory to this file, to use the other functions

from src.autokite.functions_autokite import document_top_pixels_as_pickle, find_timestamps

file_names = os.listdir("images")
files_names_path = ['images/' + f for f in file_names]
#trial_files = files_names_path[:5]
#image_paths = ['images/FE4_Image_20160901_114200_UTCp1.jpg']

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

for index, path in enumerate(files_names_path):
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
document_top_pixels_as_pickle(coords_semikite, output_file="coordinates_semikite_yx.pckl")

if clicked_coord:
    print("Saved coordinates:", clicked_coord)
else:
    print("It did not work.")


print('Finished')


# use this function:
#document_top_pixels_as_pickle