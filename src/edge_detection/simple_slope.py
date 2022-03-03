import numpy as np
from numpy import asarray
from PIL import Image, ImageOps

from src.utils.image_operations import get_horizontal_slope, get_vertical_slope, prepare_slope_array
from src.data.constants import path


selector = "armchair_at_beach"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

gs_array = asarray(gs_img)

hor_slope = np.zeros([gs_array.shape[0], gs_array.shape[1]])
ver_slope = np.zeros([gs_array.shape[0], gs_array.shape[1]])

for i in range(gs_array.shape[0]):
    for j in range(gs_array.shape[1]):
        hor_slope[i][j] = get_horizontal_slope(gs_array, i, j)
        ver_slope[i][j] = get_vertical_slope(gs_array, i, j)

total_slope = prepare_slope_array(hor_slope + ver_slope)
image_total_slope = Image.fromarray(total_slope)

image_total_slope.show()
