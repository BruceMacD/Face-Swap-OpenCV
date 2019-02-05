#! /usr/bin/env python

import cv2
import numpy as np
from constants.constants import debug_mask_cloning

POLY_FILL_COLOR = (255, 255, 255)


# get the area that was transformed in order to seamlessly clone
def calculate_mask(hull, img):
    hull_tuples = []
    # convert the lists of hull points to tuples
    for points in hull:
        hull_tuples.append((points[0], points[1]))

    # create a mask that encompasses the whole image
    mask = np.zeros(img.shape, dtype=img.dtype)

    # use the empty mask as the input image and the hull tuples a polygon vertices to fill
    # this fills only the area of the hull
    cv2.fillConvexPoly(mask, np.int32(hull_tuples), POLY_FILL_COLOR)

    hull_bounding_rectangle = cv2.boundingRect(np.float32([hull]))

    bounding_rectangle_center = (hull_bounding_rectangle[0] + int(hull_bounding_rectangle[2] / 2),
                                 hull_bounding_rectangle[1] + int(hull_bounding_rectangle[3] / 2))

    # return the mask of the face area and the center of the bounding bounding box which contains the face
    return mask, bounding_rectangle_center


# this function takes in an image with a mapped face and smooths the mask to look more natural
def merge_mask_with_image(hull, img_with_mapped_face, original_img):
    mask, center = calculate_mask(hull, original_img)
    # use seamless clone to make sure the swapped face mask looks right
    return cv2.seamlessClone(np.uint8(img_with_mapped_face), original_img, mask, center, cv2.NORMAL_CLONE)
