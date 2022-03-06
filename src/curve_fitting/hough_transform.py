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
    gs_array = asarray(gsimg)
    phis = []
    rs = []

    resolution = (180, int(gs_array.shape[1]))
    for phi in range(resolution[0]):
        for r in range(resolution[1]):
            find_edge_pixels(gs_array, phi, r, phis, rs, resolution)

        # Just for progress update purposes
        if phi == resolution[0] - 1:
            print()
        else:
            sys.stdout.write('\r' + f"Progress: {math.ceil(100 * phi / (resolution[0] - 1))} %")

    fig, ax = plt.subplots()
    ax.hexbin(phis, rs, gridsize=resolution[0])

    plt.show()

    phis_df = pd.DataFrame(phis, columns=["phis"])
    rs_df = pd.DataFrame(rs, columns=["rs"])

    edges = pd.concat([phis_df, rs_df], axis=1).reset_index(drop=True).groupby(["phis", "rs"]).size()\
        .sort_values(ascending=False)
    edges = edges[edges >= hough_threshold]

    gs_img_3d = gs_img.convert("RGB")
    gs_array_3d = asarray(gs_img_3d)

    print(edges)

    for index in edges.index:
        phi, r = index
        print(index, edges[index])
        for x in range(gs_array.shape[1]):
            if phi == 0:  # vertical line --> check all the points with x = r, than break out of the loop
                for y in range(gs_array.shape[0]):
                    gs_array_3d[y][r] = (255, 0, 0)
                break

            # if phi < 90:
            #     y_hi = int((r - x * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            #     y_lo = int((r - (x + 1) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            # else:
            #     y_lo = int((r - (x - resolution[1]) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            #     y_hi = int((r - (x + 1 - resolution[1]) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))

            y_hi = int((r - x * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))
            y_lo = int((r - (x + 1) * math.cos(math.pi * phi / 180)) / math.sin(math.pi * phi / 180))

            if y_hi < gs_array.shape[0] and y_lo >= 0:
                diff = y_hi - y_lo + 1
                for i in range(diff):
                    if i < diff / 2:
                        gs_array_3d[y_lo + i][x] = (255, 0, 0)
                    if i >= diff / 2:
                        gs_array_3d[y_lo + i][x + 1] = (255, 0, 0)

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
