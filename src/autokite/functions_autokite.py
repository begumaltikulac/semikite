"""
    Function modules that contain the cutting and blurring algorithms, as well as calculating the pixel differences
    between images.
"""


from os import listdir
from os.path import isfile, join
import pickle
import re

import cv2
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


def cutting(image: np.array) -> np.array:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = int(min(center) * 0.81)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
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

    top_coords = [[int(row), int(col)] for row, col in top_coords]

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
    for i, (y, x) in enumerate(coordinates):
        cv2.circle(highlight, (x, y), 5, (0, 0, 255), 2)
        cv2.putText(highlight, str(i + 1), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return highlight


def save_image(path: str, image: np.array, _type: str):
    return cv2.imwrite(f"{path}/{_type}_image.jpg", image)


def plot_difference_field(diff_field: np.array, title="Difference Field"):
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_field, cmap='hot')  # or 'viridis', 'plasma', 'gray'
    plt.colorbar(label='Pixel Difference')
    plt.title(title)
    plt.axis('off')
    plt.show()

    return


def plot_rgb_channel_differences(original_image: np.array, smoothed_image: np.array):
    diff = cv2.absdiff(smoothed_image, original_image)
    b_diff, g_diff, r_diff = cv2.split(diff)
    diffs = [r_diff, g_diff, b_diff]
    titles = ['Red Channel Difference', 'Green Channel Difference', 'Blue Channel Difference']

    fig, axs = plt.subplots(3, 1, figsize=(6, 18))
    cmap = 'hot'

    axs[0].imshow(r_diff, cmap=cmap)
    axs[0].set_title('Red Channel Difference')
    axs[0].axis('off')

    axs[1].imshow(g_diff, cmap=cmap)
    axs[1].set_title('Green Channel Difference')
    axs[1].axis('off')

    axs[2].imshow(b_diff, cmap=cmap)
    axs[2].set_title('Blue Channel Difference')
    axs[2].axis('off')

    for ax, data, title in zip(axs, diffs, titles):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def pixel_to_sky_angles(
        x: int,
        y: int,
        cx: int = 960,
        cy: int = 960,
        r_max: int = 960,
        projection: str = 'equidistant'
) -> (float, float):
    """
    Converts pixel coordinates (x, y) from a fisheye image to sky angles (zenith θ and azimuth φ).

    Args:
        :param x: Kite's x coordinate.
        :param y: Kite's y coordinate.
        :param cx: Fisheye camera's x coordinate.
        :param cy: Fisheye camera's y coordinate.
        :param r_max: Max radius of fisheye image.
        :param projection: Projection of fisheye image (default is "equidistant").

    Returns:
        theta: Zenith angle in degrees (0° = zenith, 90° = horizon)
        phi: Azimuth angle in degrees (0° = right/east, 90° = up/north, 180° = left/west, etc.)
    """

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Convert to zenith angle θ based on projection model
    if projection == 'equidistant':
        # r_max corresponds to θ = 90°
        theta = (r / r_max) * (np.pi / 2)
    else:
        raise NotImplementedError(f"Projection model '{projection}' not implemented.")

    # Azimuth angle φ
    phi = np.arctan2(-dy, dx)  # negative dy to match image coordinates
    phi = np.degrees(phi) % 360  # Convert to degrees and normalize

    return np.degrees(theta), phi


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