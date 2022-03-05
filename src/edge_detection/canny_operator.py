from PIL import ImageOps, Image

from src.utils.image_operations import gaussian_smoothing, sobel_grey_level_gradient, non_maxima_suppression
from src.data.variables import canny_thresholds
from src.data.constants import path


def canny_edge_detection(gsimg: Image, thresholds: list):
    print('Starting the canny operator ...')

    # smoothing image with Gaussian filter
    gs_img_smoothed = gaussian_smoothing(gsimg)
    gs_img_smoothed.show()

    # Compute grey level gradient with Sobel/Prewitt mask
    gs_array_filt = sobel_grey_level_gradient(gs_img_smoothed)

    # Apply non-maxima suppression
    gs_array_suppressed = non_maxima_suppression(gs_array_filt, thresholds)  # includes double thresholding

    return Image.fromarray(gs_array_suppressed)


selector = "lab_scrambled_2"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

img_filt = canny_edge_detection(gs_img, canny_thresholds[selector])

img_filt.show()
