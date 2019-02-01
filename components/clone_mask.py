#! /usr/bin/env python

import cv2
import numpy as np
from constants.constants import debug_mask_cloning


def calculate_mask(hull, img):
    # Calculate Mask
    hull_8U = []
    for i in range(0, len(hull)):
        hull_8U.append((hull[i][0], hull[i][1]))

    mask = np.zeros(img.shape, dtype=img.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull_8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    return mask, center


def clone_mask(warped_image, image_to_map, mask, center):
    output = cv2.seamlessClone(np.uint8(warped_image), image_to_map, mask, center, cv2.NORMAL_CLONE)

    if debug_mask_cloning:
        cv2.imshow("Face Swapped", output)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return output


def swap_mask(hull, warped_image, image_to_map):
    mask, center = calculate_mask(hull, image_to_map)
    return clone_mask(warped_image, image_to_map, mask, center)
