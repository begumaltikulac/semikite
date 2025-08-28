import pickle
import sys
import numpy as np
from collections import Counter


sys.path.append("../autokite")
with open("../autokite/coordinates.pckl", "rb") as f:
    coordinates_autokite = pickle.load(f)
with open("../semikite/coordinates_2025_08_27.pckl", "rb") as f:
        coordinates_semikite = pickle.load(f)

print(coordinates_semikite.head())     # first rows
print(coordinates_semikite.columns)    # column names
print(coordinates_semikite.index)      # index values

#isolate the timestamps
times = coordinates_autokite.index.tolist()

# delete images where we didn't see a kite in the semikite code
times_kite_found = []
for time in times:
    coords2 = coordinates_semikite.loc[time, "coordinates [y,x]"]
    x2, y2 = coords2
    if (x2 > 15) and (y2 > 15):
        times_kite_found.append(time)

# check for True and False for all times
equality = dict((time,0) for time in times)
for time in times:
    coords1 = coordinates_autokite.loc[time, "coordinates"]
    x1, y1 = coords1
    coords2 = coordinates_semikite.loc[time, "coordinates"]
    x2, y2 = coords2
    diff_x = x1 - x2
    diff_y = y1 - y2
    if (np.absolute(diff_x) < 10) & (np.absolute(diff_y) < 10):
        equality[time] = True
    else:
        equality[time] = False

# count True and False values for all times
counts = Counter(equality.values())
print(counts)

#check for True and False for all times where we clicked on a kite (in semikite)
equality_kite_found = dict((time,0) for time in times)
for time in times:
    if time in times_kite_found:
        coords1 = coordinates_autokite.loc[time, "coordinates"]
        x1, y1 = coords1
        coords2 = coordinates_semikite.loc[time, "coordinates"]
        x2, y2 = coords2
        diff_x = x1 - x2
        diff_y = y1 - y2
        if (np.absolute(diff_x) < 10) & (np.absolute(diff_y) < 10):
            equality_kite_found[time] = True
        else:
            equality_kite_found[time] = False
    else:
        equality_kite_found.pop(time, None)

# count True and False values for kite found times
counts = Counter(equality_kite_found.values())
print(counts)




