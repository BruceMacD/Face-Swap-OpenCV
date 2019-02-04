#! /usr/bin/env python
"""
Run all the separate components of face swapping in an easily understandable high-level runner class
"""

import sys
import getopt
import cv2
from components.landmark_detection import detect_landmarks
from components.convex_hull import find_convex_hull
from components.delaunay_triangulation import find_delauney_triangulation
from components.affine_transformation import apply_affine_transformation
from components.clone_mask import swap_mask

EXPECTED_NUM_IN = 2
DEBUG = True


def exit_error():
    print('Error: unexpected arguments')
    print('faceSwap.py -i <path/to/inputFile1> -i <path/to/inputFile2>')
    sys.exit()


def main(argv):
    in_imgs = []
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        exit_error()

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            in_imgs.append(arg)
        else:
            exit_error()

    # need specific number of ins
    if len(in_imgs) != EXPECTED_NUM_IN:
        exit_error()

    print('Input files', in_imgs)

    img_1 = cv2.imread(in_imgs[0])
    img_2 = cv2.imread(in_imgs[1])

    # find the facial landmarks which return the key points of the face
    # localizes and labels areas such as eyebrows and nose
    landmarks_1 = detect_landmarks(img_1)
    landmarks_2 = detect_landmarks(img_2)

    # create a convex hull around the points, this will be used for transferring the points
    # to another face
    hull_1, hull_2 = find_convex_hull(landmarks_1, landmarks_2, img_1, img_2)

    # divide the boundary of the face into smaller sections
    delauney_1 = find_delauney_triangulation(img_1, hull_1)
    delauney_2 = find_delauney_triangulation(img_2, hull_2)

    # warp the source triangles onto the target face
    # TODO: is this actually what is warped?
    img_1_warped = apply_affine_transformation(delauney_1, hull_1, hull_2, img_1, img_2)
    img_2_warped = apply_affine_transformation(delauney_2, hull_2, hull_1, img_2, img_1)

    swap_1 = swap_mask(hull_2, img_1_warped, img_2)
    swap_2 = swap_mask(hull_1, img_2_warped, img_1)

    # show the results
    cv2.imshow("Face Swap 1: ", swap_1)
    cv2.imshow("Face Swap 2: ", swap_2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
