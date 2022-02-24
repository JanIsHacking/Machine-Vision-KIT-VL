from numpy import asarray
from PIL import Image, ImageOps

from utils import *


def get_horizontal_slope(img_array, x, y):
    if 1 <= x <= img_array.shape[0] - 2:
        return (int(img_array[x + 1][y]) - int(img_array[x - 1][y])) / 2
    return 0


def get_vertical_slope(img_array, x, y):
    if 1 <= y <= img_array.shape[1] - 2:
        return (int(img_array[x][y + 1]) - int(img_array[x][y - 1])) / 2
    return 0


def prepare_slope_array(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 1:
                x[i][j] = 1

    x = np.log(x)

    x = scale_array(x, 255)
    abs_max = np.amax(np.abs(x))
    x = 255 / abs_max * x

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = int(x[i][j])

    return x


def main():
    img = Image.open("resources/rubiks_cube/scrambled_2.jpeg")

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


if __name__ == '__main__':
    main()
