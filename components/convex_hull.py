#! /usr/bin/env python
"""
A convex hull is a tight fitting boundary around points
We will we use a convex hull to find the boundary of a face given its points (obtained from openCV)
This can be made more efficient by doing multiple hulls in one iteration
Keeping simple for clarity
"""

import numpy as np
import cv2


"""
TODO: explanation
"""
def find_points(img_path):
    img_file = img_path + '.txt'
    points = []

    with open(img_file) as points_file:
        for line in points_file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


def show_convex_hull(hull, points, img_path):
    img = cv2.imread(img_path)
    # create an empty black image
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # draw points and hull
    for i in range(len(img)):
        color_points = (0, 255, 0)
        color_hull = (255, 255, 255)
        # cv2.drawKeypoints()
        cv2.drawContours(drawing, points, i, color_points, 2, 8)
        cv2.drawContours(drawing, hull, i, color_hull, 2, 8)

    cv2.imshow("Output", drawing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_convex_hull(img_path):
    hull = []
    points = find_points(img_path)
    hull_index = cv2.convexHull(np.array(points), returnPoints=False)

    for i in range(0, len(hull_index)):
        hull.append(points[int(hull_index[i])])

    # display for debugging
    # TODO: move/remove
    show_convex_hull(hull, points, img_path)
