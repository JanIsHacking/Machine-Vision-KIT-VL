from numpy import asarray
from PIL import Image, ImageOps

from utils import *


def canny_edge_detection(gs_img: Image):
    print('Starting the canny operator.')

    # smoothing image with Gaussian filter
    gs_img_smoothed = gaussian_smoothing(gs_img)
    gs_img_smoothed.show()

    # Compute grey level gradient with Sobel/Prewitt mask
    gs_array_filt = sobel_grey_level_gradient(gs_img_smoothed)

    # Apply non-maxima suppression
    gs_array_suppressed = non_maxima_suppression(gs_array_filt)  # includes double threshholding

    return Image.fromarray(gs_array_suppressed)


def main():
    scrambled2_path = "../resources/rubiks_cube/laboratory_images/top_scrambled_2.jpeg"
    armchair_patch = "../resources/armchair_at_beach.jpg"
    bottom1_path = "../resources/rubiks_cube/real_world_images/bottom_1.png"
    img = Image.open(bottom1_path)

    gs_img = ImageOps.grayscale(img)
    # gs_img.show()

    img_filt = canny_edge_detection(gs_img)

    img_filt.show()


if __name__ == '__main__':
    main()
