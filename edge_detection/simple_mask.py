import math
import sys

from PIL import Image, ImageOps

from utils import *


def main():
    img = Image.open("resources/rubiks_cube/top_scrambled_2.jpeg")

    gs_img = ImageOps.grayscale(img)
    gs_img.show()

    gs_img_filt = prewitt_grey_level_gradient(gs_img)

    gs_img_thresh = simple_threshhold(gs_img_filt, 20, 255)

    gs_img_thresh.show()


if __name__ == '__main__':
    main()
