#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Author: Bohong Li
    The automatic kite detection mechanism is based on the difference in the RGB code of the original image and
    a blurred image. The mechanism works as follows:
    - First, Gaussian smoothing is applied on the original image. The image has to be smoothed first to avoid influences
    by sharp edges between the horizon and the edge of the lense.
    - Second, a mask is defined to cut out the lense edges and the lower part of the horizon in both the smoothed and
    the original image, should the outer edge of the horizon contain buildings or obstacles.
    - Third, the difference in the RGB code between the cut-out smoothed and original is calculated. Only the red (R)
    channel is used, as the total difference leads to the results.
    - Fourth, the algorithm finds the pixel with the largest difference in the R-channel.
    - Finally, the pixel is highlighted and denoted with a 1 on the original image. The final image can also be saved.
"""

#%%

from functions_autokite import (
    cutting,
    document_top_pixels_as_pickle,
    # document_top_pixels_as_txt,
    filenames_gen,
    find_timestamps,
    find_top_pixels,
    read_image,
    rgb_calc,
    save_image,
    smoothing,
    visualize,
)

#%%
PATH = "../semikite/images"  # Change according to image folder path
# PATH = Path("Lex/test_pics")
n_pixel = 1  # number of pixel to be detected

image_filenames = filenames_gen(PATH)
timestamps = find_timestamps(image_filenames)
coords_collection = dict((time,0) for time in timestamps)
    
for original, timestamp in zip(image_filenames, timestamps):
    original = read_image(original)
    cut_original = cutting(original)
    smoothed_img = smoothing(original)
    cut_smoothed = cutting(smoothed_img)
    rgb_difference = rgb_calc(cut_original, cut_smoothed)
    pixel_coords = find_top_pixels(rgb_difference, n_pixel)
    coords_collection[timestamp] = [pixel_coords[0]]  # only for the case if one top pixel is to be found
    highlighted_pic = visualize(pixel_coords, original)
    # save_image("Lex/new_img", highlighted_pic, f"{timestamp}")
    save_image("detected_images", highlighted_pic, f"{timestamp}")

# document_top_pixels_as_txt(timestamps, coords_collection, "coordinates.txt")
document_top_pixels_as_pickle(coords_collection, output_file="coordinates.pckl")

    # plot_rgb_channel_differences(cut_original, cut_smoothed)  # COMMENT OUT IF ONE DOES NOT WANT TO PLOT THE CHANNEL DIFFERENCE