import numpy as np


prewitt_mask = np.array([[[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]],
                         [[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]]])

sobel_mask = np.array([[[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]],
                       [[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]])

gaussian_mask = np.array([[1, 4, 1],
                          [4, 19, 4],
                          [1, 4, 1]])  # sigma^2 = 0.25

laplace_mask = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 2, 1, 0],
                        [0, 2, -16, 2, 0],
                        [0, 1, 2, 1, 0],
                        [0, 0, 1, 0, 0]])

direction_lookup = ((1, -1), (0, -1), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1))

path = {
    "armchair_at_beach": "../resources/random_images/armchair_at_beach.jpg",
    "person_smiling": "../resources/random_images/person_smiling.jpg",
    "ship_full_moon": "../resources/random_images/ship_full_moon.jpg",
    "smart_building": "../resources/random_images/smart_building.jpg",
    "tiger_bird": "../resources/random_images/tiger_bird.jpg",
    "lab_scrambled_1": "../resources/rubiks_cube/laboratory_images/scrambled_1.jpeg",
    "lab_scrambled_2": "../resources/rubiks_cube/laboratory_images/scrambled_2.jpeg",
    "lab_scrambled_3": "../resources/rubiks_cube/laboratory_images/scrambled_3.jpeg",
    "lab_solved_1": "../resources/rubiks_cube/laboratory_images/solved_1.jpeg",
    "lab_solved_2": "../resources/rubiks_cube/laboratory_images/solved_2.jpeg",
    "real_bottom1": "../resources/rubiks_cube/real_world_images/scrambled_1.png"
}

canny_thresholds = {
    "armchair_at_beach": [45, 65],
    "ship_full_moon": [],
    "smart_building": [15, 30],
    "tiger_bird": [15, 25],
    "lab_scrambled_1": [],
    "lab_scrambled_2": [15, 25],
    "lab_scrambled_3": [],
    "lab_solved_1": [],
    "lab_solved_2": [],
    "real_bottom1": []
}

log_thresholds = {
    "armchair_at_beach": 25,
    "ship_full_moon": None,
    "smart_building": 15,
    "tiger_bird": None,
    "lab_scrambled_1": None,
    "lab_scrambled_2": None,
    "lab_scrambled_3": None,
    "lab_solved_1": None,
    "lab_solved_2": None,
    "real_bottom1": None
}
