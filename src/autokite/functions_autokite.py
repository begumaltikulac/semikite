"""
    Function modules that contain the cutting and blurring algorithms, as well as calculating the pixel differences
    between images.
"""


from os import listdir
from os.path import isfile, join
import pickle
import re

import cv2
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filenames_gen(path: str) -> list:
    image_names = [f for f in listdir(path) if isfile(join(path, f))]
    original_images = []

    for name in image_names:
        original_images.append(f"{path}/{name}")
        original_images.sort()

    return original_images


def find_timestamps(img_names: list):
    times = []

    for fname in img_names:
        match = re.search(r'\d{8}_\d{6}', fname)
        if match:
            times.append(match.group())

    return times


def read_image(image: str):
    pic = cv2.imread(image)
    return pic


def smoothing(image: np.array):
    smoothed_pic = cv2.GaussianBlur(image, (69, 69), 30)

    return smoothed_pic


def cutting(image: np.array, radius_frac: float = 0.8, top_fraction: float = 0.15) -> np.array:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = int(min(center) * radius_frac)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create circular mask
    cv2.circle(mask, center, radius, 255, -1)
    # Mask out top part
    top_cutoff = int(height * top_fraction)
    cv2.rectangle(mask, (0, 0), (width, top_cutoff), 0, -1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def rgb_calc(original_image: np.array, smoothed_image: np.array):
    diff = cv2.absdiff(smoothed_image, original_image)  # shape (H, W, 3)
    total_diff = diff[:, :, 0]

    return total_diff


def find_top_pixels(total_diff: np.array, top_n: int) -> list:
    flat = total_diff.flatten()
    top_indices = np.argpartition(flat, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(-flat[top_indices])]  # Sort descending
    top_coords = [np.unravel_index(_index, total_diff.shape) for _index in top_indices]

    top_coords = [[int(col), int(row)] for row, col in top_coords]  # [x,y]

    return top_coords


def document_top_pixels_as_txt(times: list, coords: dict, output_file: str) -> None:
    with open(output_file, "w"):
        pass
    with open(output_file, "a") as f:
        for time in times:
            if isinstance(coords[time], int):
                continue
            else:
                f.write(f"Kite at {time}: \n")
                for c in coords[time]:
                    f.write(f"{c}\n")
            f.write(f"\n")


def document_top_pixels_as_pickle(coords: dict, output_file: str) -> None:
    df_coordinates = pd.DataFrame(
        {"time": coords.keys(), "coordinates [x,y]": coords.values()},
    ).set_index("time")
    with open(output_file, "wb") as f:
        pickle.dump(df_coordinates, f)
    return


def visualize(coordinates: np.array, original_image: np.array):
    highlight = original_image.copy()
    for i, (x, y) in enumerate(coordinates):
        cv2.circle(highlight, (x, y), 5, (0, 0, 255), 2)
        cv2.putText(highlight, str(i + 1), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return highlight


def save_image(path: str, image: np.array, _type: str):
    return cv2.imwrite(f"{path}/{_type}_image.jpg", image)


def pixel_to_sky_angles(
    x,
    y,
    cx=960,
    cy=960,
    r_max=960
):
    """
    Convert fisheye pixel (x,y) to (elevation, azimuth) using your diagram’s rings.

    Elevation is derived directly from radial distance:
      - 90° at center (zenith),
      - 0° at rim (horizon).
    Azimuth matches your compass:
      - 0° = W, 90° = N, 180° = E, 270° = S.

    Parameters
    ----------
    x, y : int or float
        Pixel coordinates.
    cx, cy : int or float
        Image center (default: 960,960 for 1920×1920).
    r_max : float
        Radius from center to horizon (default: 960).

    Returns
    -------
    elevation_deg : float
        Elevation angle in degrees, rounded to 2 decimals.
    azimuth_deg : float
        Azimuth angle in degrees, rounded to 2 decimals.
    """
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)

    # --- Elevation from ring position ---
    theta = 90.0 * (1 - (r / r_max))

    # Ingo Lange's correction polynomial for the elevation angle 
    theta = -6.380024219e-7*theta**4 + 1.384399783e-4*theta**3 - 1.122405179e-2*theta**2 + 1.326190211*theta+2.494295303

    # Azimuth angle φ
    phi = np.degrees(np.arctan2(-dy, dx)) % 360  # negative dy to match image coordinates

    return round(theta, 2), round(phi, 2)


def pixel_to_angles_with_height(
        x: int,
        y: int,
        altitude: float,
        cx: int = 960,
        cy: int = 960,
        r_max: int = 960,
        projection: str = 'equidistant',
) -> (float, float, (float, float, float)):
    """
    Convert a fisheye pixel coordinate into real-world coordinates at a known altitude.

    This function assumes the camera is pointing upward (towards the sky) and uses a
    fisheye projection model to map pixel coordinates into a 3D direction vector.
    Given the real-world altitude of the object (above the camera), the function extends
    the ray until it intersects the horizontal plane at that altitude.

    Parameters
    ----------
    x : float
        X-coordinate (column index) of the pixel in the image (in pixels).
    y : float
        Y-coordinate (row index) of the pixel in the image (in pixels).
    cx : float
        X-coordinate of the fisheye image center (in pixels).
    cy : float
        Y-coordinate of the fisheye image center (in pixels).
    r_max : float
        Radius of the fisheye projection circle in pixels.
        This corresponds to a zenith angle of 90° (the horizon).
    altitude : float
        Known altitude of the object above the camera (in meters).
    projection : str, optional
        Fisheye projection model. Currently only supports "equidistant".
        Default is "equidistant".

    Returns
    -------
    theta : float
        Zenith angle of the object in degrees.
        - 0° = directly overhead (zenith, positive Z-axis).
        - 90° = on the horizon.
    phi : float
        Azimuth angle of the object in degrees, in the range 0 to 359.
        - 0° = east (right in image coordinates).
        - 90° = north (up in image coordinates).
        - 180° = west (left).
        - 270° = south (down).
    position : tuple of float
        Real-world Cartesian coordinates (X, Y, Z) of the object in **meters**
        relative to the camera origin (0, 0, 0).
        - X points east (right).
        - Y points north (up).
        - Z points up (zenith).
        The returned Z value will equal the given `altitude`.

    Raises
    ------
    ValueError
        If the computed ray points below the horizon (vz <= 0),
        making altitude intersection impossible.

    Notes
    -----
    - Input values are in **pixels** for image coordinates and **meters** for altitude.
    - Output angles are in **degrees**, and output coordinates are in **meters**.
    - The camera is assumed to be located at the origin (0,0,0) with its optical
      axis pointing upward (positive Z).
    - No lens distortion correction is applied. For high-precision results,
      intrinsic calibration parameters should be used (e.g., OpenCV fisheye model).

    """
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)

    # Pixel distance -> zenith angle θ
    if projection == 'equidistant':
        theta = (r / r_max) * (np.pi / 2)  # 0..90°
    else:
        raise NotImplementedError(f"Projection '{projection}' not implemented.")

    # Azimuth φ
    phi = np.arctan2(-dy, dx)  # negative dy for image coords
    phi_deg = np.degrees(phi) % 360

    # Direction vector from camera
    vx = np.cos(phi) * np.sin(theta)
    vy = np.sin(phi) * np.sin(theta)
    vz = np.cos(theta)

    if vz <= 0:
        raise ValueError("Object ray points below horizon. Altitude intersection impossible.")

    # Scale ray so Z = altitude
    scale = altitude / vz
    new_x = vx * scale
    new_y = vy * scale
    new_z = vz * scale  # should equal altitude

    # Recompute corrected zenith (since now we know real-world Z)
    r_xy = np.sqrt(new_x**2 + new_y**2)
    theta_corr = np.degrees(np.arctan2(r_xy, new_z))

    return int(theta_corr), int(phi_deg), (new_x, new_y, new_z)


def check_false_detection(pixel_file: str, mean_deviation: int, y_dev_threshold: int) -> pd.DataFrame:
    with open(pixel_file, "rb") as f:
        blab = pickle.load(f)

    x = []
    y = []
    for element in blab["coordinates [x,y]"]:
        x.append(element[0])
        y.append(element[1])
    x = np.array(x)
    y = np.array(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_deviation = x - x_mean
    y_deviation = y - y_mean

    blab["x_deviation"] = x_deviation
    blab["y_deviation"] = y_deviation
    blab["valid"] = (y > y_dev_threshold) | ((abs(x_deviation) < mean_deviation) & (abs(y_deviation) < mean_deviation))
    #  y_dev_threshold checks in which part of the image the kite is. This applies to when the wind direction is known
    #  apriori

    return blab[blab["valid"]==False]


def open_theodolite(file: str, obs_date: str, start_time: str) -> pd.DataFrame:
    start_time = pd.to_datetime(f"{obs_date} {start_time}")

    # Read all lines first
    with open(file, "r") as f:
        theo_all = f.readlines()
    # Skip last three lines
    # theo_all = theo_all[:-3]

    time_sec = []
    azimuth = []
    elevation = []

    for line in theo_all:
        line = line.strip()
        if line.startswith("D") or line.startswith("E"):  # data line
            parts = line.split()
            time_sec.append(float(parts[1]))
            azimuth.append(float(parts[2]))
            elevation.append(float(parts[3]))
        # elif line.startswith("S"):
            # print("Metadata:", line)  # optional

    df_theo = pd.DataFrame({
        "time_sec": [start_time + timedelta(seconds=s) for s in time_sec],
        "azimuth": azimuth,
        "elevation": elevation
    })

    return df_theo