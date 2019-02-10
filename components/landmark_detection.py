#! /usr/bin/env python
"""
Using the provided functions in dlib to detect the points of facial landmarks in an image
"""
import numpy as np
import cv2
import dlib
from constants.constants import debug_landmark_detection

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

    # return the array of (x, y)-coordinates
    return coords


def show_face_annotated(faces, landmarks, img):

    for face in faces:
        # draw box for face
        x, y, w, h = dlib_to_cv_bounding_box(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw circles for landmarks
        for landmark_set in landmarks:
            for x, y in landmark_set:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_landmarks(img):
    # this list will contain the facial landmark points for each face detected
    points = []
    # second argument of 1 indicates the image will be upscaled once, upscaling creates a bigger image so it is easier
    # to detect the faces, can increase this number if there are troubles detecting faces
    # returns a bounding box around each face
    detected_faces = frontal_face_detector(img, 1)

    # now that we have the boxes containing the faces find the landmarks inside them
    for face in detected_faces:
        # use dlib to find the expected facial landmarks in the boxes around the detected faces
        landmarks = landmarks_predictor(img, face)
        # add the facial landmarks in a form we can use later without dlib
        points.append(landmarks_to_numpy(landmarks))

    # show the bounding box
    if debug_landmark_detection:
        show_face_annotated(detected_faces, points, img)

    return points
