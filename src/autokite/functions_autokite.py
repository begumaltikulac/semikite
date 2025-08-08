"""
    Function modules that contain the cutting and blurring algorithms, as well as calculating the pixel differences
    between images.
"""

from os import listdir
from os.path import isfile, join
import re

import cv2
import numpy as np
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

    return top_coords


def document_top_pixels(times: list, coords: dict, output_file: str) -> None:
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