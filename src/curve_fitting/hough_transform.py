import numpy as np
import pandas as pd
import math
from datetime import datetime
import sys
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from numpy import asarray

from src.data.constants import path
from src.data.variables import hough_thresholds


def find_edge_pixels(gs_array: np.ndarray, phi: int, r: int, phis: list, rs: list, resolution):
    for x in range(resolution[1] - 1):
        if phi == 0:  # vertical line --> check all the points with x = r, than break out of the loop
            for y in range(gs_array.shape[0]):
                if gs_array[y][r] == 255:
                    phis.append(phi)
                    rs.append(r)
            break

        if phi < 90:
            y_hi = int((r - x * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            y_lo = int((r - (x + 1) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
        else:
            y_lo = int((r - (x - resolution[1]) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            y_hi = int((r - (x + 1 - resolution[1]) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))

        if y_hi < gs_array.shape[0] and y_lo >= 0:
            diff = y_hi - y_lo + 1
            for i in range(diff):
                if i < diff / 2 and gs_array[y_lo + i][x] == 255 or (i >= diff / 2 and gs_array[y_lo + i][x + 1] == 255):
                    phis.append(phi)
                    rs.append(r)


def hough_transform(gsimg: Image, hough_threshold: int):
    # Convert the input grayscale image to a NumPy array
    gs_array = asarray(gsimg)

    # Lists to store detected phis and rs
    phis = []
    rs = []

    # Define the resolution for the Hough space
    resolution = (180, int(gs_array.shape[1]))

    # Loop over phi and r values to populate the Hough space
    for phi in range(resolution[0]):
        for r in range(resolution[1]):
            # Find edge pixels and accumulate phis and rs
            find_edge_pixels(gs_array, phi, r, phis, rs, resolution)

        # Print progress update
        if phi == resolution[0] - 1:
            print()
        else:
            sys.stdout.write('\r' + f"Progress: {math.ceil(100 * phi / (resolution[0] - 1))} %")

    # Create a hexbin plot for visualization
    fig, ax = plt.subplots()
    ax.hexbin(phis, rs, gridsize=resolution[0])
    plt.show()

    # Convert phis and rs to DataFrames for analysis
    phis_df = pd.DataFrame(phis, columns=["phis"])
    rs_df = pd.DataFrame(rs, columns=["rs"])

    # Group by phis and rs, count occurrences, and filter by the Hough threshold
    edges = pd.concat([phis_df, rs_df], axis=1).reset_index(drop=True).groupby(["phis", "rs"]).size() \
        .sort_values(ascending=False)
    edges = edges[edges >= hough_threshold]

    # Convert the original grayscale image to RGB
    gs_img_3d = gsimg.convert("RGB")
    gs_array_3d = asarray(gs_img_3d)

    # Print the detected edges
    print(edges)

    # Highlight detected lines in the image
    for index in edges.index:
        phi, r = index
        print(index, edges[index])
        for x in range(gs_array.shape[1]):
            if phi == 0:  # vertical line --> check all the points with x = r, than break out of the loop
                for y in range(gs_array.shape[0]):
                    gs_array_3d[y][r] = (255, 0, 0)
                break

            # Calculate y_hi and y_lo for non-vertical lines
            y_hi = int((r - x * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            y_lo = int((r - (x + 1) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))

            # Check bounds and highlight pixels
            if y_hi < gs_array.shape[0] and y_lo >= 0:
                diff = y_hi - y_lo + 1
                for i in range(diff):
                    if i < diff / 2:
                        gs_array_3d[y_lo + i][x] = (255, 0, 0)
                    if i >= diff / 2:
                        gs_array_3d[y_lo + i][x + 1] = (255, 0, 0)

    # Return the RGB image with highlighted edges
    return gs_img_3d


selector = "headphones_case"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

time_1 = sum([a * b for a, b in zip([int(x) for x in datetime.now().strftime("%X").split(":")],
                                    [3600, 60, 1])])

result = hough_transform(gs_img, hough_thresholds[selector])

time_2 = sum([a * b for a, b in zip([int(x) for x in datetime.now().strftime("%X").split(":")],
                                    [3600, 60, 1])])

print(f"This only took {round((time_2 - time_1) / 60, 1)} minutes")

result.show()
