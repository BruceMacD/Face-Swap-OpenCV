#! /usr/bin/env python
"""
A convex hull is a tight fitting boundary around points
We will we use a convex hull to find the boundary of a face given its points (obtained from openCV)
This can be made more efficient by doing multiple hulls in one iteration
Keeping simple for clarity
"""

import numpy as np
import cv2
from constants.constants import debug_convex_hull


def show_convex_hull(hull, img):
    # draw points within the hull
    for x, y in hull:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_convex_hull(points_1, points_2, img_1, img_2):
    hull_1 = []
    hull_2 = []

    # this is the area that we will be mapping between faces
    hull_index_to_map = cv2.convexHull(np.array(points_1), returnPoints=False)

    # find the facial landmark points on both faces that are within the hull of the face we are basing our map off of
    for i in range(0, len(hull_index_to_map)):
        hull_1.append(points_1[int(hull_index_to_map[i])])
        hull_2.append(points_2[int(hull_index_to_map[i])])

    # display for debugging
    if debug_convex_hull:
        show_convex_hull(hull_1, img_1)
        show_convex_hull(hull_2, img_2)

    return hull_1, hull_2
