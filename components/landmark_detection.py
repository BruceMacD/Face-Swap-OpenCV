#! /usr/bin/env python
"""
Using the provided functions in dlib to detect the points of facial landmarks in an image
"""
import numpy as np
import cv2
import dlib
from constants.constants import debug

# Pre-trained shape predictor from iBUG 300-W dataset
SHAPE_PREDICTOR = 'data/shape_predictor_68_face_landmarks.dat'

frontal_face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


# convenience function from imutils
def dlib_to_cv_bounding_box(box):
    # convert dlib bounding box for OpenCV display
    x = box.left()
    y = box.top()
    w = box.right() - x
    h = box.bottom() - y

    return x, y, w, h


# another conversion from imutils
def landmarks_to_numpy(landmarks):
    # initialize the matrix of (x, y)-coordinates with a row for each landmark
    coords = np.zeros((landmarks.num_parts, 2), dtype=int)

    # convert each landmark to (x, y)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def show_face_annotated(faces, landmarks, img):
    # boxes_landmarks = zip(faces, landmarks)

    for face in faces:
        # draw box for face
        x, y, w, h = dlib_to_cv_bounding_box(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw circles for landmarks
        for x, y in landmarks:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_landmarks(img):
    points = []
    # second argument of 1 indicates the image will be upscaled once
    # returns a bounding box around each face
    detected_faces = frontal_face_detector(img, 1)

    # now that we have the boxes containing the faces find the landmarks inside them
    for face in detected_faces:
        # we are assuming that we will only find one face here
        # in any case, we can only swap one face for our use so we will be taking the last one found
        landmarks = landmarks_predictor(img, face)
        points = landmarks_to_numpy(landmarks)

    # show the bounding box
    if debug:
        show_face_annotated(detected_faces, points, img)

    return points
