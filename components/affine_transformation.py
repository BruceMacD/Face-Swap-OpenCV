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

    # if debug_affine_transformation:
    #     cv2.imshow("Affine transformation", dst)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()

    return dst


def morph_triangular_regions(triangles_1, triangles_2, img_1, img_2):
    # Find bounding rectangle for each triangle
    rectangle_1 = cv2.boundingRect(np.float32([triangles_1]))
    rectangle_2 = cv2.boundingRect(np.float32([triangles_2]))

    # x, y, w, h = cv2.boundingRect(np.float32([triangles_1]))
    # if debug_affine_transformation:
    #     # cv2.imshow("Mask for transformation", rectangle_2)
    #     cv2.rectangle(img_1, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #     # cv2.rectangle()
    #     cv2.imshow("Rectangle", img_1)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()

    # Offset points by left top corner of the respective rectangles
    triangles_rectangle_1 = []
    triangles_rectangle_2 = []
    # TODO: this might not be needed, just convert triangles_rectangle_2
    triangles_rectangle_2_int = []

    for i in range(0, 3):
        triangles_rectangle_1.append(((triangles_1[i][0] - rectangle_1[0]), (triangles_1[i][1] - rectangle_1[1])))
        triangles_rectangle_2.append(((triangles_2[i][0] - rectangle_2[0]), (triangles_2[i][1] - rectangle_2[1])))
        triangles_rectangle_2_int.append(((triangles_2[i][0] - rectangle_2[0]), (triangles_2[i][1] - rectangle_2[1])))

    # Get mask by filling triangle
    # mask = np.zeros((rectangle_2[3], rectangle_2[2], 3), dtype=np.float32)
    mask = np.zeros((rectangle_2[3], rectangle_2[2], 3))
    cv2.fillConvexPoly(mask, np.int32(triangles_rectangle_2_int), (1.0, 1.0, 1.0), 16, 0)

    # if debug_affine_transformation:
    #     cv2.imshow("Mask before transformation", mask)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()

    # Apply warpImage to small rectangular patches
    img_1_rect = img_1[rectangle_1[1]:rectangle_1[1] + rectangle_1[3], rectangle_1[0]:rectangle_1[0] + rectangle_1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (rectangle_2[2], rectangle_2[3])

    img_2_rect = apply_affine_transform(img_1_rect, triangles_rectangle_1, triangles_rectangle_2, size)

    img_2_rect = img_2_rect * mask

    # if debug_affine_transformation:
    #     cv2.imshow("Mask for transformation", img_2_rect)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()

    # Copy triangular region of the rectangular patch to the output image
    img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] = \
        img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] \
        * ((1.0, 1.0, 1.0) - mask)

    img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] \
        = img_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]] \
        + img_2_rect

    return img_2


# TODO: remove img_1 and img_2 from params after debugging
def get_corresponding_triangles(delauney, hull_1, hull_2, img_1, img_2):
    # triangles_1 = []
    # triangles_2 = []
    # line_color = (255, 255, 255)
    img_1_warped = np.copy(img_2)

    for i in range(0, len(delauney)):

        triangles_1 = []
        triangles_2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            triangles_1.append(hull_1[delauney[i][j]])
            triangles_2.append(hull_2[delauney[i][j]])

        morph_triangular_regions(triangles_1, triangles_2, img_1, img_1_warped)

        # if debug_affine_transformation:
        #     # draw the triangle on the picture for the case we want to display it
        #     a = tuple(hull_1[delauney[i][0]])
        #     b = tuple(hull_1[delauney[i][1]])
        #     c = tuple(hull_1[delauney[i][2]])
        #
        #     cv2.line(img_1, a, b, line_color, 1, cv2.LINE_AA, 0)
        #     cv2.line(img_1, b, c, line_color, 1, cv2.LINE_AA, 0)
        #     cv2.line(img_1, c, a, line_color, 1, cv2.LINE_AA, 0)

    # if debug_affine_transformation:
    #     cv2.imshow("Affine transformation", img_1)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()

    return img_1_warped


def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    morphed_img = get_corresponding_triangles(delauney, hull_1, hull_2, img_1, img_2)

    # morphed_img = morph_triangular_regions(triangles_1, triangles_2, img_1, img_1_warped)

    if debug_affine_transformation:
        cv2.imshow("Affine transformation", morphed_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return morphed_img
