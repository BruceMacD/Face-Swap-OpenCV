#! /usr/bin/env python
"""
Using the provided functions in dlib to detect the points of facial landmarks in an image
"""
import numpy as np
import argparse
import cv2
import dlib
from constants.constants import debug

frontal_face_detector = dlib.get_frontal_face_detector()


# convenience function from imutils
def dlib_to_cv_bounding_box(box):
    # convert dlib bounding box for OpenCV display
    x = box.left()
    y = box.top()
    w = box.right() - x
    h = box.bottom() - y

    return x, y, w, h


def show_face_bb(faces, img):
    for face in faces:
        x, y, w, h = dlib_to_cv_bounding_box(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        cv2.waitKey(0)


def detect_landmarks(img):
    # second argument of 1 indicates the image will be upscaled once
    detected_faces = frontal_face_detector(img, 1)
    # show the bounding box
    if debug:
        show_face_bb(detected_faces, img)

