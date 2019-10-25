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


def predict_next(previous, current):
    prev_rotation = previous[:3, :3]
    prev_translation = previous[:, 3:4]
    cur_rotation = current[:3, :3]
    cur_translation = current[:, 3:4]
    rotation_predict = np.dot(cur_rotation, np.dot(cur_rotation, np.linalg.inv(prev_rotation)))
    translation_predict = cur_translation + cur_translation - prev_translation
    predicted = np.hstack((rotation_predict, translation_predict))
    return predicted
