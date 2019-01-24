#! /usr/bin/env python
"""
Convert the facial landmark points on the plane into triangles
These triangles allow us to morph the face to fit on another face
"""

import cv2
from constants.constants import debug_delauney_triangulation


line_color = (255, 255, 255)


# if the x or y cords are outside the x/y cords of the bounding rectangle return false
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

        # the lines between points that make up the triangle (the sides of the triangle)
        a = (triangle[0], triangle[1])
        b = (triangle[2], triangle[3])
        c = (triangle[4], triangle[5])

        # only add triangles that are within the bounds of the image
        if inside_rect_bounds(a, rectangle) and inside_rect_bounds(b, rectangle) and inside_rect_bounds(c, rectangle):
            point_indices = []

            # find if there is a <x,y> point roughly approximate for each point of the triangle in our points
            # if there is an <x,y> point corresponding add the index of the point for use
            for i in range(0, len(points)):
                # compare a.x to point.x and a.y to point.y etc
                if abs(a[0] - points[i][0]) < 1.0 and abs(a[1] - points[i][1]) < 1.0:
                    point_indices.append(i)
                if abs(b[0] - points[i][0]) < 1.0 and abs(b[1] - points[i][1]) < 1.0:
                    point_indices.append(i)
                if abs(c[0] - points[i][0]) < 1.0 and abs(c[1] - points[i][1]) < 1.0:
                    point_indices.append(i)

            # if each point in the triangle has a corresponding point store the indices to retrieve the <x,y> points
            # we use these indices later to get the corresponding points from another hull/landmarks
            if len(point_indices) == 3:
                delauney_triangulation.append((point_indices[0], point_indices[1], point_indices[2]))
                # draw the triangle on the picture for the case we want to display it
                cv2.line(img, a, b, line_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, b, c, line_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, c, a, line_color, 1, cv2.LINE_AA, 0)

    if debug_delauney_triangulation:
        cv2.imshow("Delauney Triangulation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return delauney_triangulation
