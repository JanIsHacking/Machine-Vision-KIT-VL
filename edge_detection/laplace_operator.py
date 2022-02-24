from PIL import ImageOps

from utils import *


def laplacian_of_gaussian_edge_detection(gs_img: Image, threshold: int):
    print('Starting the LoG edge detection ...')

    gs_img_filt = laplacian_of_gaussian(gs_img)

    # simple_threshold_testing(gs_img_filt)

    gs_img_thresh = simple_threshold(gs_img_filt, threshold, 255)

    return gs_img_thresh


def main():
    selector = "smart_building"
    img = Image.open(path[selector])

    gs_img = ImageOps.grayscale(img)
    gs_img.show()

    img_filt = laplacian_of_gaussian_edge_detection(gs_img, log_thresholds[selector])

    img_filt.show()


if __name__ == '__main__':
    main()
