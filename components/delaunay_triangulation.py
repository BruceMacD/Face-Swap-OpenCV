#! /usr/bin/env python
"""
Convert the facial landmark points on the plane into triangles
These triangles allow us to morph the face to fit on another face
"""

import cv2
from constants.constants import debug_delauney_triangulation


line_color = (255, 255, 255)


def inside_rect_bounds(point, rectangle):
    x = point[0]
    y = point[1]

    if x < rectangle[0] or x > rectangle[2]:
        return False
    if y < rectangle[1] or y > rectangle[3]:
        return False
    return True


# we are using the convex hull but this also works given the facial landmark points
def find_delauney_triangulation(img, points):
    size_img = img.shape
    # find the space we want to partition, in this case the image size
    rectangle = (0, 0, size_img[1], size_img[0])

    # create a subdivision for triangulation
    subdivision = cv2.Subdiv2D(rectangle)
    # populate the subdivision with our facial landmark points
    for point in points:
        subdivision.insert(tuple(point))

    delauney_triangulation = []

    # create triangles from the points in the subdivision and display them
    for triangle in subdivision.getTriangleList():

        # the lines between points that make up the triangle
        a = (triangle[0], triangle[1])
        b = (triangle[2], triangle[3])
        c = (triangle[4], triangle[5])

        if inside_rect_bounds(a, rectangle) and inside_rect_bounds(b, rectangle) and inside_rect_bounds(c, rectangle):
            delauney_triangulation.append((a, b, c))
            cv2.line(img, a, b, line_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, b, c, line_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, c, a, line_color, 1, cv2.LINE_AA, 0)

    if debug_delauney_triangulation:
        cv2.imshow("Delauney Triangulation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return delauney_triangulation
