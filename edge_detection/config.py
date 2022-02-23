import numpy as np


prewitt_mask = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]], [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])
sobel_mask = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
gaussian_mask = np.array([[1, 4, 1], [4, 19, 4], [1, 4, 1]])  # sigma^2 = 0.25

direction_lookup = ((1, -1), (0, -1), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1))

# scrambled 2 --> (5, 20)
# armchair --> (45, 65)

lower_threshhold = 15
upper_threshhold = 30
