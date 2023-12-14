import math

from PIL import Image
import numpy as np
from datetime import datetime

from src.data.constants import path
from src.data.variables import ccl_thresholds


def connected_components_labeling(c_img: Image, threshold: float):
    # Convert the input image to a NumPy array
    c_array = np.array(c_img)

    # Create an empty array to store segment labels, initialized with -1
    seg_array = np.empty((c_array.shape[0], c_array.shape[1]), dtype=int)
    seg_array.fill(-1)

    # List to store individual segments
    segments = []

    # Iterate through each pixel in the image
    for u in range(c_array.shape[1]):
        for v in range(c_array.shape[0]):
            # Calculate the absolute difference between the current pixel and its left neighbor
            if u == 0:
                ln_cd = math.inf
            else:
                ln_cd = sum([abs(int(a) - int(b)) for a, b in zip(c_array[v, u], c_array[v, u - 1])])

            # Calculate the absolute difference between the current pixel and its upper neighbor
            if v == 0:
                un_cd = math.inf
            else:
                un_cd = sum([abs(int(a) - int(b)) for a, b in zip(c_array[v, u], c_array[v - 1, u])])

            # Check if the left and upper neighbors are below the threshold
            if ln_cd > threshold:
                if un_cd > threshold:
                    # Create a new segment if both neighbors are above the threshold
                    segments.append([(u, v)])
                    seg_array[v, u] = len(segments) - 1
                else:
                    # Assign the current pixel to the segment of its upper neighbor
                    seg_array[v, u] = seg_array[v - 1, u]
                    segments[seg_array[v, u]].append((u, v))
            else:
                if un_cd > threshold:
                    # Assign the current pixel to the segment of its left neighbor
                    seg_array[v, u] = seg_array[v, u - 1]
                    segments[seg_array[v, u]].append((u, v))
                else:
                    if seg_array[v - 1, u] == seg_array[v, u - 1]:
                        # Merge segments if the left and upper neighbors belong to different segments
                        seg_array[v, u] = seg_array[v, u - 1]
                        segments[seg_array[v, u]].append((u, v))
                    else:
                        # Merge left neighbor pixels into the upper neighbor segment
                        temp_list = segments[seg_array[v, u - 1]]
                        segments[seg_array[v, u - 1]] = []
                        segments[seg_array[v, u - 1]].append((u, v))
                        for x, y in segments[seg_array[v, u - 1]]:
                            seg_array[y, x] = seg_array[v - 1, u]
                        segments[seg_array[v - 1, u]] = temp_list + segments[seg_array[v - 1, u]]

    # Assign the final color to each pixel based on the color of the first pixel in its segment
    for segment in segments:
        for x, y in segment:
            c_array[y, x] = c_array[segment[0][1], segment[0][0]]

    # Convert the NumPy array back to an image and return
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
