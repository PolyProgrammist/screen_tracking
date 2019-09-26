import math
import cv2
import numpy as np
import pywavefront
import logging
import transforms3d
from scipy.spatial import distance

np.set_printoptions(precision=3, suppress=True)


def to_screen(point):
    return tuple(map(int, point))

# Find rotation and translation
_, rotation, translation = cv2.solvePnP(model_vertices, first_frame_pixels, camera_params, np.array([]))

# Find external parameters matrix
R_rodrigues = cv2.Rodrigues(rotation)[0]
external_matrix = np.hstack((R_rodrigues, translation))

points = screen_points(camera_params, external_matrix, model_vertices)

# Reading first video frame
cap = cv2.VideoCapture(video_source)
ret, img = cap.read()

for i, point in enumerate(points):
    cv2.circle(img, to_screen(point), 3, (0, 0, 255), -1)
    cv2.line(img, to_screen(points[i]), to_screen(points[(i + 1) % len(points)]), (0, 0, 255))

cv2.imshow('First frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
