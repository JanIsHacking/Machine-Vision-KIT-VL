import math
import time

from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
from numpy import asarray

from src.data.constants import path
from src.data.variables import ccl_thresholds


def connected_components_labeling(c_img: Image, threshold: float):
    c_array = asarray(c_img)
    seg_array = np.empty((c_array.shape[0], c_array.shape[1]), dtype=int)
    seg_array.fill(-1)

    segments = []

    for u in range(c_array.shape[1]):
        for v in range(c_array.shape[0]):
            # assigning the left neighbor
            if u == 0:
                ln_cd = math.inf
            else:
                ln_cd = sum([abs(int(a) - int(b)) for a, b in zip(c_array[v, u], c_array[v, u - 1])])

            # assigning the upper neighbor
            if v == 0:
                un_cd = math.inf
            else:
                un_cd = sum([abs(int(a) - int(b)) for a, b in zip(c_array[v, u], c_array[v - 1, u])])

            if ln_cd > threshold:
                if un_cd > threshold:
                    segments.append([(u, v)])
                    seg_array[v, u] = len(segments) - 1
                else:
                    seg_array[v, u] = seg_array[v - 1, u]
                    segments[seg_array[v, u]].append((u, v))
            else:
                if un_cd > threshold:
                    seg_array[v, u] = seg_array[v, u - 1]
                    segments[seg_array[v, u]].append((u, v))
                else:
                    if seg_array[v - 1, u] == seg_array[v, u - 1]:
                        seg_array[v, u] = seg_array[v, u - 1]
                        segments[seg_array[v, u]].append((u, v))
                    else:
                        # merge left neighbor pixels into upper neighbor segment
                        temp_list = segments[seg_array[v, u - 1]]
                        segments[seg_array[v, u - 1]] = []
                        segments[seg_array[v, u - 1]].append((u, v))
                        for x, y in segments[seg_array[v, u - 1]]:
                            seg_array[y, x] = seg_array[v - 1, u]
                        segments[seg_array[v - 1, u]] = temp_list + segments[seg_array[v - 1, u]]

    for segment in segments:
        for x, y in segment:
            c_array[y, x] = c_array[segment[0][1], segment[0][0]]
    return Image.fromarray(c_array)


selector = "lab_scrambled_2"
img = Image.open(path[selector])

time_1 = sum([a * b for a, b in zip([int(x) for x in datetime.now().strftime("%X").split(":")],
                                    [3600, 60, 1])])

seg_img = connected_components_labeling(img, ccl_thresholds[selector])

time_2 = sum([a * b for a, b in zip([int(x) for x in datetime.now().strftime("%X").split(":")],
                                    [3600, 60, 1])])

print(f"This only took {round(time_2 - time_1, 5)} seconds")

seg_img.show()
