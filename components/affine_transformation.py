#! /usr/bin/env python
"""
Morph the triangulation of one face onto another
Maps corresponding triangles between faces
"""

import cv2
import numpy as np
from constants.constants import debug_delauney_triangulation


def morph_triangular_regions(triangles_1, triangles_2, img_1, img_2):
    # Find bounding rectangle for each triangle
    rectangle_1 = cv2.boundingRect(np.float32([triangles_1]))
    rectangle_2 = cv2.boundingRect(np.float32([triangles_2]))

    # Offset points by left top corner of the respective rectangles
    triangle_rectangles_1 = []
    triangle_rectangles_2 = []
    # t2RectInt = []

    for i in range(0, 3):
        triangle_rectangles_1.append(((triangles_1[i][0] - rectangle_1[0]), (triangles_1[i][1] - rectangle_1[1])))
        triangle_rectangles_2.append(((triangles_2[i][0] - rectangle_2[0]), (triangles_2[i][1] - rectangle_2[1])))


def get_corresponding_triangles(delauney, hull_1, hull_2):
    triangles_1 = []
    triangles_2 = []

    for i in range(0, len(delauney)):

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            triangles_1.append(hull_1[delauney[i][j]])
            triangles_2.append(hull_2[delauney[i][j]])

    return triangles_1, triangles_2


def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    triangles_1, triangles_2 = get_corresponding_triangles(delauney, hull_1, hull_2)

    morph_triangular_regions(triangles_1, triangles_2, img_1, img_2)
    return
