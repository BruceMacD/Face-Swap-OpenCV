#! /usr/bin/env python
"""
Morph the triangulation of one face onto another
Maps corresponding triangles between faces
"""

import cv2
import numpy as np
from constants.constants import debug_delauney_triangulation


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morph_triangular_regions(triangles_1, triangles_2, img_1, img_2):
    # Find bounding rectangle for each triangle
    rectangle_1 = cv2.boundingRect(np.float32([triangles_1]))
    rectangle_2 = cv2.boundingRect(np.float32([triangles_2]))

    # Offset points by left top corner of the respective rectangles
    triangles_rectangle_1 = []
    triangles_rectangle_2 = []
    # t2RectInt = []

    for i in range(0, 3):
        triangles_rectangle_1.append(((triangles_1[i][0] - rectangle_1[0]), (triangles_1[i][1] - rectangle_1[1])))
        triangles_rectangle_2.append(((triangles_2[i][0] - rectangle_2[0]), (triangles_2[i][1] - rectangle_2[1])))

    # Get mask by filling triangle
    mask = np.zeros((rectangle_2[3], rectangle_2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(triangles_rectangle_2), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img_1[rectangle_1[1]:rectangle_1[1] + rectangle_1[3], rectangle_1[0]:rectangle_1[0] + rectangle_1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (rectangle_2[2], rectangle_2[3])

    img2Rect = applyAffineTransform(img1Rect, triangles_rectangle_1, triangles_rectangle_2, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] = \
        img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] \
        * ((1.0, 1.0, 1.0) - mask)

    img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] \
        = img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] + img2Rect

    return img_2


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

    return morph_triangular_regions(triangles_1, triangles_2, img_1, img_2)
