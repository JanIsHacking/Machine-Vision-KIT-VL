import numpy as np
import math
import sys

from src.data.constants import direction_lookup


def mask_convolution(gs_array: np.ndarray, mask: np.ndarray):
    hor_bw = int(mask.shape[0] / 2)  # horizontal border width
    ver_bw = int(mask.shape[1] / 2)  # vertical border width

    limits = np.array([[hor_bw, gs_array.shape[0] - 1 - hor_bw],
                       [ver_bw, gs_array.shape[1] - 1 - ver_bw]])

    gs_array_filt = np.zeros([limits[0][1] - limits[0][0], limits[1][1] - limits[1][0]])

    # Applying the mask to every pixel
    for i in range(limits[0][0], limits[0][1]):
        for j in range(limits[1][0], limits[1][1]):
            gs_array_filt[i - limits[0][0]][j - limits[1][0]] \
                = apply_mask(gs_array[i - hor_bw:i + 1 + hor_bw, j - ver_bw:j + 1 + ver_bw], mask)

        # Just for progress update purposes
        if i == limits[0][1] - 1:
            print()
        else:
            sys.stdout.write('\r' + f"Progress: {math.ceil(100 * i / limits[0][1])} %")

    return gs_array_filt


def apply_mask(gs_arr: np.ndarray, mask: np.ndarray):
    total_value = 0
    abs_sum = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            total_value += gs_arr[mask.shape[1] - 1 - j][mask.shape[0] - 1 - i] * mask[j][i]
            abs_sum += abs(mask[i][j])

    return total_value / abs_sum


def scale_array(x: np.ndarray, y: int):
    abs_max = np.amax(np.abs(x))
    return y / abs_max * x


def radian_to_position(radians: float):
    sl = 2 * math.pi / 8
    radians += sl / 2  # adding half of the slice to the radians to rotate the axes in order for them to
    # match the neightbor directions, apply int for 'easier calculation' purposes
    return direction_lookup[int(radians - radians % sl / sl) % 8]


def calculate_direction(hor: float, ver: float):
    if hor == 0:
        if ver == 0:  # origin
            return 0
        if ver > 0:  # up
            return math.pi / 2
        return 3 / 2 * math.pi  # down
    elif hor > 0:
        if ver == 0:  # right
            return 0
        if ver > 0:  # first quadrant
            return math.atan(ver / hor)
        return 3 / 2 * math.pi + math.atan(hor / abs(ver))  # forth quadrant
    else:
        if ver == 0:  # left
            return math.pi
        if ver >= 0:  # second quadrant
            return 1 / 2 * math.pi + math.atan(abs(hor) / ver)
        return math.pi + math.atan(abs(hor) / abs(ver))  # third quadrant
