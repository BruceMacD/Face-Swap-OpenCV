#! /usr/bin/env python
"""
Run all the separate components of face swapping in an easily understandable high-level runner class
"""

import sys
import getopt
import cv2
from constants.constants import debug
from components.landmark_detection import detect_landmarks

EXPECTED_NUM_IN = 2
DEBUG = True


def exit_error():
    print('Error: unexpected arguments')
    print('faceSwap.py -i <path/to/inputFile1> -i <path/to/inputFile2> -o <outputDirectory>')
    sys.exit()


def main(argv):
    in_imgs = []
    out_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        exit_error()
    # need exactly the expected parameters
    if len(opts) != 3:
        exit_error()

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            in_imgs.append(arg)
        elif opt in ("-o", "--ofile"):
            out_dir = arg

    # last check, need specific number of ins and one out
    if len(in_imgs) != EXPECTED_NUM_IN and len(out_dir == ''):
        exit_error()

    print('Input files', in_imgs)
    print('Output file is', out_dir)

    img1 = cv2.imread(in_imgs[0])
    img2 = cv2.imread(in_imgs[1])

    detect_landmarks(img1)
    detect_landmarks(img2)

    # TODO: Testing image display, should be removed
    # cv2.imshow('image', img1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # Find the facial landmarks which return points on the face

    # TODO: convex hull from points
    # find_convex_hull(in_imgs[0])


if __name__ == "__main__":
    main(sys.argv[1:])
