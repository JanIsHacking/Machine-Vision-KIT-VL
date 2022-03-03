from PIL import ImageOps, Image

from src.utils.image_operations import laplacian_of_gaussian, simple_threshold
from src.data.variables import log_thresholds
from src.data.constants import path


def laplacian_of_gaussian_edge_detection(gsimg: Image, threshold: int):
    print('Starting the LoG edge detection ...')

    gs_img_filt = laplacian_of_gaussian(gsimg)

    # simple_threshold_testing(gs_img_filt)

    gs_img_thresh = simple_threshold(gs_img_filt, threshold, 255)

    return gs_img_thresh


selector = "armchair_at_beach"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

img_filt = laplacian_of_gaussian_edge_detection(gs_img, log_thresholds[selector])

img_filt.show()
