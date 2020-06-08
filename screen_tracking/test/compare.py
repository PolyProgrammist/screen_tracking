import logging

import numpy as np
import transforms3d
from scipy.spatial import distance

from screen_tracking.common import normalize_angle, screen_points

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
    logging.info(title + ':')
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


# TODO: divide on rotation_translation and points and move to common/common.py
def rotation_translation_points(extrinsic, projection, vertices):
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:, 3:4]
    points = screen_points(projection, extrinsic, vertices)
    return rotation, translation, points


def external_difference(actual, expected, projection, model_vertices):
    norm_coefficient = np.linalg.norm(model_vertices[0] - model_vertices[2])  # screen diagonal
    actual_rotation, actual_translation, actual_points = \
        rotation_translation_points(actual, projection, model_vertices)
    expected_rotation, expected_translation, expected_points = \
        rotation_translation_points(expected, projection, model_vertices)

    rotation_diff = rotation_difference(actual_rotation, expected_rotation)
    translation_diff = translation_difference(actual_translation, expected_translation, norm_coefficient)
    points_diff = points_difference(actual_points, expected_points, norm_coefficient)

    return np.array([rotation_diff, translation_diff, points_diff])


def compare(model_vertices, projection_matrix, ground_truth, tracking_result, frames_to_compare=None):
    untracked_frames = []
    average_diff = np.array([0.0, 0.0, 0.0])
    for frame, ground_truth_matrix in ground_truth.items():
        if frame in tracking_result:
            current_diff = external_difference(tracking_result[frame]['pose'], ground_truth_matrix, projection_matrix, model_vertices)
            log_errors(current_diff, 'Frame ' + str(frame))
            if frames_to_compare is None or frame in frames_to_compare:
                average_diff += current_diff
        else:
            untracked_frames.append(frame)

    frames_number = len(ground_truth.items()) - len(untracked_frames)
    average_diff /= frames_number if not frames_to_compare else len(frames_to_compare)
    log_errors(average_diff, 'Average diff')
    if untracked_frames:
        logging.error('Untracked frames: ' + str(untracked_frames))
    else:
        logging.info('No untracked frames')
    return average_diff

def compare_one(algorithm, kwargs, reader, steps):
    if not 'compare' in steps:
        return

    model_vertices, projection_matrix, ground_truth, tracking_result = reader.compare_input()
    if 'broken' in tracking_result:
        return 'broken'

    comparing_result = {}
    for frame, ground_truth_matrix in ground_truth.items():
        current_diff = external_difference(tracking_result[frame]['pose'], ground_truth_matrix, projection_matrix,
                                           model_vertices)
        rotation_diff = current_diff[0]
        translation_diff = current_diff[1]

        comparing_result[frame] = {}
        comparing_result[frame]['time'] = tracking_result[frame]['time']
        comparing_result[frame]['rotation_diff'] = rotation_diff
        comparing_result[frame]['translation_diff'] = translation_diff

    return comparing_result

def compare_many(function_results, algorithm):
    print(algorithm)
    rotation_dif_all = 0
    translation_dif_all = 0
    all = 0
    passed = 0
    passed_translation = 0
    passed_rotation = 0
    frame_handle_time = {}
    all_not_broken = {}
    translation_difmax = 0.02
    rotation_difmax = 0.02
    for dir, elem in function_results.items():
        if elem['result'] == 'broken':
            # TODO: change to number of frames
            all += 10 if 'generated_tv_on9' in dir or 'generated_tv_off9' in dir else 9
            continue
        for frame, result in elem['result'].items():
            if frame == 1:
                continue
            rotation_dif = result['rotation_diff']
            translation_dif = result['translation_diff']
            rotation_dif_all += rotation_dif
            translation_dif_all += translation_dif
            passed += translation_dif <= translation_difmax and rotation_dif <= rotation_difmax
            passed_translation += translation_dif <= translation_difmax
            passed_rotation += rotation_dif <= rotation_difmax
            all += 1

            if elem['suffix'] not in frame_handle_time:
                frame_handle_time[elem['suffix']] = 0
                all_not_broken[elem['suffix']] = 0
            all_not_broken[elem['suffix']] += 1
            frame_handle_time[elem['suffix']] += result['time']

    times = []
    for suffix in frame_handle_time:
        times.append(frame_handle_time[suffix] / all_not_broken[suffix])
    print('FPS mean:', np.mean(times))
    print('FPS variance:', np.std(times))

    rotation_dif_all /= all
    translation_dif_all /= all
    print('Tracking success rate:', passed / all)
    print(passed_rotation, passed_translation)