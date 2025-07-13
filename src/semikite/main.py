import matplotlib
matplotlib.use('TkAgg')  # <- important command for GUI window

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = ['FE4_Image_20160901_115500_UTCp1.jpg']

clicked_coord = []

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