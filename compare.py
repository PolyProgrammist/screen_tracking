import numpy as np
import transforms3d
from scipy.spatial import distance
import logging, coloredlogs
from enum import Enum

from common import normalize_angle, screen_points
from utils import TrackingDataReader

DIFFERENCE_PARAMETERS = [
    (0.04, 'External parameters rotation'),
    (0.1, 'External parameters translation'),
    (0.1, 'Projected points')
]


def log_error_if_differ_much(diff, threshold, name):
    external_differ_output = name + ' differ by: ' + '%.4f' % diff
    if diff > threshold:
        logging.error(external_differ_output)
    else:
        logging.info(external_differ_output)


def log_errors(difference, title):
    print(title + ':')
    for threshold, name, diff in zip(*zip(*DIFFERENCE_PARAMETERS), difference):
        log_error_if_differ_much(diff, threshold, name)


def rotation_difference(actual, expected):
    diff_matrix = np.dot(actual, np.linalg.inv(expected))
    diff_direction, diff_angle = transforms3d.axangles.mat2axangle(diff_matrix)
    diff_angle = normalize_angle(diff_angle)
    return diff_angle


def translation_difference(actual, expected, norm_coefficient):
    diff = distance.euclidean(actual, expected)
    diff /= norm_coefficient
    return diff


def points_difference(actual, expected, norm_coefficient):
    diff = np.linalg.norm(actual - expected, axis=1).mean()
    diff /= norm_coefficient
    return diff


def rotation_translation_points(extrinsic, projection, vertices):
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:, 3:4]
    points = screen_points(projection, extrinsic, vertices)
    return rotation, translation, points


def difference(actual, expected, projection, model_vertices):
    norm_coefficient = np.linalg.norm(model_vertices[0] - model_vertices[2])  # screen diagonal
    actual_rotation, actual_translation, actual_points = \
        rotation_translation_points(actual, projection, model_vertices)
    expected_rotation, expected_translation, expected_points = \
        rotation_translation_points(expected, projection, model_vertices)

    rotation_diff = rotation_difference(actual_rotation, expected_rotation)
    translation_diff = translation_difference(actual_translation, expected_translation, norm_coefficient)
    points_diff = points_difference(actual_points, expected_points, norm_coefficient)

    return np.array([rotation_diff, translation_diff, points_diff])


def compare(model_vertices, projection_matrix, ground_truth, tracking_result):
    untracked_frames = []
    average_diff = np.array([0.0, 0.0, 0.0])
    for key, value in ground_truth.items():
        if key in tracking_result:
            tr = tracking_result[key]
            gt = value
            current_diff = difference(tr, gt, projection_matrix, model_vertices)
            log_errors(current_diff, 'Frame ' + str(key))
            average_diff += current_diff
        else:
            untracked_frames.append(key)

    frames_number = len(ground_truth.items()) - len(untracked_frames)
    average_diff /= frames_number
    log_errors(average_diff, 'Average diff')

    logging.error('Untracked frames: ' + str(untracked_frames))


if __name__ == "__main__":
    coloredlogs.install()
    reader = TrackingDataReader()
    compare(*reader.compare_input())
