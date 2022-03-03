from PIL import Image
from copy import copy
import numpy as np

from image_operations import simple_threshold, double_threshold


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