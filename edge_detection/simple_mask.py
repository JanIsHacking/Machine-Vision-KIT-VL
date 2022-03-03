from PIL import ImageOps

from utils import *


img = Image.open("resources/rubiks_cube/scrambled_2.jpeg")

gs_img = ImageOps.grayscale(img)
gs_img.show()

gs_img_filt = prewitt_grey_level_gradient(gs_img)

gs_img_thresh = simple_threshold(gs_img_filt, 20, 255)

gs_img_thresh.show()
