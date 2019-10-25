import cv2
import numpy as np

from screen_tracking.common.common import screen_points
from screen_tracking.test.compare import external_difference


def get_external_matrix(tracker, points):
    _, rotation, translation = cv2.solvePnP(tracker.model_vertices, points, tracker.camera_params,
                                            np.array([]), flags=tracker.tracker_params.PNP_FLAG)
    R_rodrigues = cv2.Rodrigues(rotation)[0]
    external_matrix = np.hstack((R_rodrigues, translation))
    return external_matrix


def get_screen_points(tracker, external_matrix):
    points = screen_points(tracker.camera_params, external_matrix, tracker.model_vertices)
    return points


def external_matrices_difference(tracker, points):
    return external_difference(
        get_external_matrix(tracker, points),
        tracker.last_matrix,
        tracker.camera_params,
        tracker.model_vertices
    )
