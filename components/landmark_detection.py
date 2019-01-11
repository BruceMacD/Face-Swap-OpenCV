#! /usr/bin/env python
"""
Using the provided functions in dlib to detect the points of facial landmarks,
"""
import numpy as np
import argparse
import cv2
import dlib
from constants.constants import debug


def detect(img, debug):
    _debug = debug
    print("detecting")
