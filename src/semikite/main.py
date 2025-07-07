import matplotlib
matplotlib.use('TkAgg')  # <- wichtig für GUI-Fenster

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('FE4_Image_20160901_115500_UTCp1.jpg')

clicked_coord = None

def onclick(event):
    global clicked_coord
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        clicked_coord = (x, y)
        print(f"Geklickt bei: x={x}, y={y}")
        plt.close()

fig, ax = plt.subplots()
ax.imshow(img)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Bitte ins Bild klicken")
plt.show()

if clicked_coord:
    print("Gespeicherte Koordinaten:", clicked_coord)
else:
    print("Kein Klick erkannt.")