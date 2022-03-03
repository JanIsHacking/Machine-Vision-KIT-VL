from PIL import Image
import numpy as np
import sys
from numpy import asarray, amax, amin
from numpy.linalg import eig
import math

from src.utils.general_purpose import mask_convolution, scale_array, calculate_direction, radian_to_position
from src.data.constants import gaussian_mask, prewitt_mask, sobel_mask, laplace_mask
from src.data.constants import direction_lookup


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


def get_horizontal_slope(img_array, x, y):
    if 1 <= x <= img_array.shape[0] - 2:
        return (int(img_array[x + 1][y]) - int(img_array[x - 1][y])) / 2
    return 0


def get_vertical_slope(img_array, x, y):
    if 1 <= y <= img_array.shape[1] - 2:
        return (int(img_array[x][y + 1]) - int(img_array[x][y - 1])) / 2
    return 0


def prepare_slope_array(x):
    for k in range(x.shape[0]):
        for m in range(x.shape[1]):
            if x[k][m] < 1:
                x[k][m] = 1

    x = np.log(x)

    x = scale_array(x, 255)
    abs_max = np.amax(np.abs(x))
    x = 255 / abs_max * x

    for k in range(x.shape[0]):
        for m in range(x.shape[1]):
            x[k][m] = int(x[k][m])

    return x


def horizontal_derivative(gs_array: np.ndarray):
    deriv = np.zeros((gs_array.shape[0] - 2, gs_array.shape[1] - 2))
    for i in range(0, deriv.shape[0]):
        for j in range(0, deriv.shape[1]):
            deriv[i][j] = (int(gs_array[i + 2][j + 1]) - int(gs_array[i][j + 1])) / 2
    return deriv


def vertical_derivative(gs_array: np.ndarray):
    deriv = np.zeros((gs_array.shape[0] - 2, gs_array.shape[1] - 2))
    for i in range(0, deriv.shape[0]):
        for j in range(0, deriv.shape[1]):
            deriv[i][j] = (int(gs_array[i + 1][j + 2]) - int(gs_array[i + 1][j])) / 2
    return deriv


def diss_eig(hor_deriv: np.ndarray, ver_deriv: np.ndarray, x, y):
    ret = np.zeros((2, 2))
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            ret[0, 0] += math.pow(hor_deriv[i][j], 2)
            ret[0, 1] += hor_deriv[i][j] + ver_deriv[i][j]
            ret[1, 0] += hor_deriv[i][j] + ver_deriv[i][j]
            ret[1, 1] += math.pow(ver_deriv[i][j], 2)
    return eig(ret)[0]


def calculate_dissimilarity(gs_img: Image, theta: float, alpha: float):
    gs_array = asarray(gs_img)  # n x m
    hor_deriv = horizontal_derivative(gs_array)   # (n - 1) x (m - 1)
    ver_deriv = vertical_derivative(gs_array)  # (n - 1) x (m - 1)
    result = np.zeros((gs_array.shape[0] - 4, gs_array.shape[1] - 4))  # (n - 2) x (m - 2)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            eigvals = diss_eig(hor_deriv, ver_deriv, i + 1, j + 1)
            lam1 = amax(eigvals)
            lam2 = amin(eigvals)
            if lam1 + lam2 < theta:
                result[i][j] = 0  # neither corner, nor edge
            elif lam2 > alpha * lam1:
                result[i][j] = 2  # corner
            else:
                result[i][j] = 1  # edge

        # Just for progress update purposes
        if i == result.shape[0] - 1:
            print()
        else:
            sys.stdout.write('\r' + f"Progress: {math.ceil(100 * i / result.shape[0])} %")
    return result
