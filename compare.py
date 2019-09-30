import numpy as np
import transforms3d
from scipy.spatial import distance
import logging

from common import normalize_angle, screen_points
from utils import TrackingDataReader


def log_error_if_differ_much(diff, threshold, name):
    external_differ_output = name + ' differ by: ' + '%.4f' % diff
    if diff > threshold:
        logging.error(external_differ_output)
    else:
        print(external_differ_output)


def rotation_difference(actual, expected):
    diff_matrix = np.dot(actual, np.linalg.inv(expected))
    diff_direction, diff_angle = transforms3d.axangles.mat2axangle(diff_matrix)
    diff_angle = normalize_angle(diff_angle)
    log_error_if_differ_much(diff_angle, 0.04, 'External parameters rotation')
    return diff_angle


def translation_difference(actual, expected, norm_coefficient):
    diff = distance.euclidean(actual, expected)
    diff /= norm_coefficient
    log_error_if_differ_much(diff, 0.1, 'External parameters translation')
    return diff


def points_difference(actual, expected, norm_coefficient):
    diff = np.linalg.norm(actual - expected, axis=1).mean()
    diff /= norm_coefficient
    log_error_if_differ_much(diff, 0.1, 'Projected points')
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
    rotation_difference(actual_rotation, expected_rotation)
    translation_difference(actual_translation, expected_translation, norm_coefficient)
    points_difference(actual_points, expected_points, norm_coefficient)


reader = TrackingDataReader()
untracked_frames = []
for key, value in reader.get_ground_truth().items():
    if key in reader.get_tracking_result():
        print('Frame:', key)
        tr = reader.get_tracking_result()[key]
        gt = value
        difference(tr, gt, reader.get_projection_matrix(), reader.get_model_vertices())
    else:
        untracked_frames.append(key)

logging.error('Untracked frames: ' + str(untracked_frames))