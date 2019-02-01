#! /usr/bin/env python
"""
Morph the triangulation of one face onto another
Maps corresponding triangles between faces
"""

import cv2
import numpy as np
from constants.constants import debug_affine_transformation


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morph_triangular_region(triangles_1, triangles_2, img_1, img_2):
    # Find bounding rectangle for each triangle
    x_1, y_1, w_1, h_1 = cv2.boundingRect(np.float32([triangles_1]))
    x_2, y_2, w_2, h_2 = cv2.boundingRect(np.float32([triangles_2]))

    # Offset points by left top corner of the respective rectangles
    triangles_rectangle_1 = []
    triangles_rectangle_2 = []

    for i in range(0, 3):
        triangles_rectangle_1.append(((triangles_1[i][0] - x_1), (triangles_1[i][1] - y_1)))
        triangles_rectangle_2.append(((triangles_2[i][0] - x_2), (triangles_2[i][1] - y_2)))

    # Get mask by filling triangle
    mask = np.zeros((h_2, w_2, 3))
    cv2.fillConvexPoly(mask, np.int32(triangles_rectangle_2), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img_1_rect = img_1[y_1:y_1 + h_1, x_1:x_1 + w_1]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (w_2, h_2)

    img_2_rect = apply_affine_transform(img_1_rect, triangles_rectangle_1, triangles_rectangle_2, size)

    img_2_rect = img_2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] * ((1.0, 1.0, 1.0) - mask)

    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] + img_2_rect

    return img_2


def morph_corresponding_triangles(delauney, hull_1, hull_2, img_1, img_2):
    img_1_warped = np.copy(img_2)

    for i in range(0, len(delauney)):

        triangles_1 = []
        triangles_2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            triangles_1.append(hull_1[delauney[i][j]])
            triangles_2.append(hull_2[delauney[i][j]])

        morph_triangular_region(triangles_1, triangles_2, img_1, img_1_warped)

    if debug_affine_transformation:
        cv2.imshow("Affine transformation", img_1)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return img_1_warped


def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    return morph_corresponding_triangles(delauney, hull_1, hull_2, img_1, img_2)
