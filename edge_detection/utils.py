import math
import sys
from copy import copy

from numpy import asarray
from PIL import Image

from config import *


# expects an image as input, returns an image
def gaussian_smoothing(gs_img: Image):
    print('Starting Gaussian smoothing ...')

    gs_array = asarray(gs_img)
    return Image.fromarray(mask_convolution(gs_array, gaussian_mask))


# expects an image as input, returns a 3D array with the following axes
# x: pixel position in horizontal direction
# y: pixel position in vertical direction
# z: gradient value [0] and gradient direction (in radians) [1]
def prewitt_grey_level_gradient(gs_img: Image):
    gs_array = asarray(gs_img)

    mask_hor = prewitt_mask[0]
    mask_ver = prewitt_mask[1]

    print('Starting horizontal prewitt filtering ...')
    hor = mask_convolution(gs_array, mask_hor)
    print('Starting vertical prewitt filtering ...')
    ver = mask_convolution(gs_array, mask_ver)

    Image.fromarray(scale_array(hor + ver, 255)).show()

    gs_array_filt = calculate_gradient(hor, ver)

    return gs_array_filt


# expects an image as input, returns a 3D array with the following axes
# x: pixel position in horizontal direction
# y: pixel position in vertical direction
# z: gradient value [0] and gradient direction (in radians) [1]
def sobel_grey_level_gradient(gs_img: Image):
    gs_array = asarray(gs_img)

    mask_hor = sobel_mask[0]
    mask_ver = sobel_mask[1]

    print('Starting horizontal sobel filtering ...')
    hor = mask_convolution(gs_array, mask_hor)
    print('Starting vertical sobel filtering ...')
    ver = mask_convolution(gs_array, mask_ver)

    gs_array_filt = calculate_gradient(hor, ver)

    return gs_array_filt


def calculate_gradient(hor: np.ndarray, ver: np.ndarray):
    gradient_array = np.zeros([hor.shape[0], hor.shape[1], 2])

    for i in range(hor.shape[0]):
        for j in range(hor.shape[1]):
            gradient_array[i][j][0] = hor[i][j] + ver[i][j]  # gradient value
            gradient_array[i][j][1] = calculate_direction(hor[i][j], ver[i][j])  # gradient direction in radians

    return gradient_array


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


def non_maxima_suppression(gs_array: np.ndarray, thresholds: list):
    print('Starting non maxima suppression calculation ...')

    # Taking absolute value of gradient to simplify furhter calculations
    for i in range(gs_array.shape[0]):
        for j in range(gs_array.shape[1]):
            gs_array[i][j][0] = abs(gs_array[i][j][0])

    limits = np.array([[1, gs_array.shape[0] - 2],
                       [1, gs_array.shape[1] - 2]])

    gs_array_filt = np.zeros([limits[0][1] - limits[0][0], limits[1][1] - limits[1][0]])
    for i in range(1, gs_array_filt.shape[0] - 2):
        for j in range(1, gs_array_filt.shape[1] - 2):
            hor, ver = radian_to_position(gs_array[i][j][1])
            gs_value_fp = gs_array[i + hor][j + ver][0]
            gs_value_bp = gs_array[i - hor][j - ver][0]
            if gs_array[i][j][0] > gs_value_bp and gs_array[i][j][0] > gs_value_fp:
                gs_array_filt[i][j] = gs_array[i][j][0]
            else:
                gs_array_filt[i][j] = 0
    gs_array_filt = scale_array(gs_array_filt, 255)

    # double_threshold_testing(gs_array_filt)

    double_threshold(gs_array_filt, thresholds)

    return gs_array_filt


def radian_to_position(radians: float):
    sl = 2 * math.pi / 8
    radians += sl / 2  # adding half of the slice to the radians to rotate the axes in order for them to
    # match the neightbor directions, apply int for 'easier calculation' purposes
    return direction_lookup[int(radians - radians % sl / sl) % 8]


def double_threshold(gs_array: np.ndarray, thresholds: list):
    print('Starting double threshold calculation ...')

    unsures = []

    for i in range(1, gs_array.shape[0] - 1):
        for j in range(1, gs_array.shape[1] - 1):
            if gs_array[i][j] < thresholds[0]:
                gs_array[i][j] = 0
            elif gs_array[i][j] > thresholds[1]:
                gs_array[i][j] = 1
            else:
                gs_array[i][j] = -1
                unsures.append((i, j))

    gs_array = check_unsures(unsures, gs_array)

    for i in range(1, gs_array.shape[0] - 1):
        for j in range(1, gs_array.shape[1] - 1):
            if gs_array[i][j] == -1:
                gs_array[i][j] = 0

    gs_array *= 255

    return gs_array


def check_unsures(unsures: list, gs_array: np.ndarray):
    change_counter = 0  # measures, if at least one pixel has been converted this iteration
    for unsure in unsures:
        uhor, uver = unsure
        check_sum = 0  # measures, how many neighboring pixels are surely or unsurely edge pixels
        for direction in direction_lookup:
            hor, ver = direction
            check_sum += math.pow(gs_array[uhor - hor][uver - ver], 2)
            if gs_array[uhor - hor][uver - ver] == 1:
                gs_array[uhor][uver] = 1
                unsures.remove(unsure)
                change_counter += 1
                break
        if check_sum == 0:  # check sum is only 0, if all neighbor pixels are 0, hence surely NOT an edge pixel
            unsures.remove(unsure)

    if change_counter > 0:
        gs_array = check_unsures(unsures, gs_array)

    return gs_array


def laplacian_of_gaussian(gs_img: Image):
    gs_array = asarray(gs_img)

    gs_array_filt = mask_convolution(gs_array, laplace_mask)

    for i in range(gs_array_filt.shape[0]):
        for j in range(gs_array_filt.shape[1]):
            gs_array_filt[i][j] = abs(gs_array_filt[i][j])

    return Image.fromarray(gs_array_filt)


# expects an image as input, value of threshold and grey level value to set if a pixel exceeds the threshold
def simple_threshold(gs_img: Image, y: int, z: int):
    x = asarray(gs_img)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < y:
                x[i][j] = 0
            else:
                x[i][j] = z
    return Image.fromarray(x)


###########################
# General purpose functions
###########################


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


###################
# Testing functions
###################

def simple_threshold_testing(gs_img: Image):
    print('Starting simple threshold testing ...')

    test_thresholds = []

    start = 5
    end = 60
    width = 5

    for i in range(start, end, width):
        test_thresholds.append(i)

    for threshold in test_thresholds:
        simple_threshold(gs_img, threshold, 255).show()


def double_threshold_testing(gs_array: np.ndarray):
    print('Starting double threshold testing calculation ...')

    local_thresholds = []

    starting_value = 5
    widths = [10, 20, 25]
    number = 5

    print("Testing thresholds:")
    for width in widths:
        for i in range(1, number + 1):
            start = starting_value + (i - 1) * width
            local_thresholds.append([start, start + width])
            if i % number == number - 1:
                print(local_thresholds[i * number:])

    for threshold in local_thresholds:
        Image.fromarray(double_threshold(copy(gs_array), threshold)).show()
