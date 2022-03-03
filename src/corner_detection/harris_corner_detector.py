from PIL import Image, ImageOps
import numpy as np
from numpy import asarray

from src.utils.image_operations import calculate_dissimilarity
from src.data.constants import path
from src.data.variables import harris_alphas, harris_thetas


def harris_corner_detection(gsimg: Image, theta: float, alpha: float):
    diss = calculate_dissimilarity(gsimg, theta, alpha)
    gs_corners = np.dstack(3 * [asarray(gsimg)])
    gs_edges = np.dstack(3 * [asarray(gsimg)])

    for i in range(diss.shape[0]):
        for j in range(diss.shape[1]):
            if diss[i][j] == 2:  # corner
                gs_corners[i][j] = np.array([255, 0, 0])  # pixel is turned red
            elif diss[i][j] == 1:  # edge
                gs_edges[i][j] = np.array([0, 0, 255])  # pixel is turned blue
    return Image.fromarray(gs_corners), Image.fromarray(gs_edges)


selector = "lab_solved_2"
img = Image.open(path[selector])

gs_img = ImageOps.grayscale(img)
gs_img.show()

corners, edges = harris_corner_detection(gs_img, harris_thetas[selector], harris_alphas[selector])

corners.show()
edges.show()
