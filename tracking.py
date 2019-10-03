import math
import cv2
import numpy as np
import pywavefront
import logging
import transforms3d
from scipy.spatial import distance

from utils import TrackingDataReader
from common import screen_points

np.set_printoptions(precision=3, suppress=True)

reader = TrackingDataReader()

model_vertices, camera_params, video_source, frame_pixels, result_file = reader.tracker_input()

tracking_result = {}
for frame, pixels in frame_pixels.items():
    _, rotation, translation = cv2.solvePnP(model_vertices, pixels, camera_params, np.array([]))
    R_rodrigues = cv2.Rodrigues(rotation)[0]
    external_matrix = np.hstack((R_rodrigues, translation))
    tracking_result[frame] = external_matrix

reader.write_result(tracking_result)

# cap = cv2.VideoCapture(video_source)
