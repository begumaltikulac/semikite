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
    check_false_detection,
    cutting,
    document_top_pixels_as_pickle,
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
DATE = "20250901"
pixel_threshold = 250
y_threshold = 1000
n_pixel = 1  # number of pixel to be detected
radius_cut = 0.87
top_cut = 0.2

for time_measured in ["morning", "afternoon"]:
    PATH = f"images_{DATE}/{time_measured}"  # Change according to image folder path
    coords_outfile = f"coordinates/{DATE}/coordinates_{DATE}_{time_measured}.pckl"
    detection_outfile = f"coordinates/{DATE}/false_detection_{DATE}_{time_measured}.csv"
    image_filenames = filenames_gen(PATH)
    timestamps = find_timestamps(image_filenames)
    coords_collection = dict((time, 0) for time in timestamps)

    for original, timestamp in zip(image_filenames, timestamps):
        original = read_image(original)
        cut_original = cutting(original, radius_frac=radius_cut, top_fraction=top_cut)
        smoothed_img = smoothing(original)
        cut_smoothed = cutting(smoothed_img, radius_frac=radius_cut, top_fraction=top_cut)
        rgb_difference = rgb_calc(cut_original, cut_smoothed)
        pixel_coords = find_top_pixels(rgb_difference, n_pixel)
        coords_collection[timestamp] = pixel_coords[0]  # only for the case if one top pixel is to be found
        highlighted_pic = visualize(pixel_coords, original)
        save_image(f"LEX_detected_images/{DATE}", highlighted_pic, f"{timestamp}")

    document_top_pixels_as_pickle(coords_collection, output_file=coords_outfile)
    df_false_detection = check_false_detection(coords_outfile, mean_deviation=pixel_threshold,
                                               y_dev_threshold=y_threshold)
    df_false_detection.to_csv(
        detection_outfile,
    )