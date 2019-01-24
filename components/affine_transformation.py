#! /usr/bin/env python
"""
Morph the triangulation of one face onto another
Maps corresponding triangles between faces
"""

import cv2
import numpy as np
from constants.constants import debug_delauney_triangulation


def apply_affine_transformation(delauney, hull_1, hull_2):
    for i in range(0, len(delauney)):
        triangles_1 = []
        triangles_2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            triangles_1.append(hull_1[delauney[i][j]])
            triangles_2.append(hull_2[delauney[i][j]])

        # warpTriangle(img1, img1Warped, t1, t2)
    return
