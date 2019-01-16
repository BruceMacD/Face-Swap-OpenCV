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
    color = (255, 255, 255)  # color for convex hull
    height, width, channels = img.shape
    drawing = np.zeros((width, height, 3), np.uint8)
    # convert the hull to Numpy array of (x,y) points for display
    hull_contour = np.array(hull).reshape((-1, 1, 2)).astype(np.int32)

    # for i in range(len(hull)):
    #     cv2.drawContours(img, hull_contour, i, color, 2, 8)

    # draw points within the hull
    for x, y in hull:
        cv2.circle(drawing, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Output", drawing)
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
