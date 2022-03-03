from PIL import ImageOps, Image

from src.utils.image_operations import prewitt_grey_level_gradient, simple_threshold
from src.data.constants import path


selector = "lab_scrambled_2"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

gs_img_filt = prewitt_grey_level_gradient(gs_img)

gs_img_thresh = simple_threshold(gs_img_filt, 20, 255)

gs_img_thresh.show()
