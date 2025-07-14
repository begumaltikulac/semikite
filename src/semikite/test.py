import os
import matplotlib
matplotlib.use('TkAgg')  # Important for interactive GUI

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ✅ 1. Set the folder where your images are
image_folder = 'C:/Users/BEGUM/PycharmProjects/semikite/images/'

# ✅ 2. Automatically collect all .jpg image paths
image_paths = [os.path.join(image_folder, f)
               for f in sorted(os.listdir(image_folder))
               if f.endswith('.jpg')]

clicked_coord = []

def onclick(event):
    global clicked_coord
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        clicked_coord.append((x, y))
        print(f"Coordinates: x={x}, y={y}")
        plt.close()

# ✅ 3. Display each image one by one
for path in image_paths:
    print(f"Opening: {path}")
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(15, 25))
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Please click on the kite in the image")
    plt.show()

if clicked_coord:
    print("Saved coordinates:", clicked_coord)
else:
    print("It did not work.")
