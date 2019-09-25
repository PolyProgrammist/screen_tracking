import math
import cv2
import numpy as np
import pywavefront
import logging
import transforms3d
from scipy.spatial import distance

model_vertices_file = 'sources/tv_picture_centered.obj'
camera_params_file = 'sources/camera_params.txt'
first_frame_pixels_file = 'sources/first_frame_2d_pixels.txt'
external_parameters_ground_truth = 'sources/result_object_animation.txt'
video_source = 'sources/tv_on.mp4'

np.set_printoptions(precision=3, suppress=True)


def read_camera_params():
    lines = open(camera_params_file).readlines()
    for i, line in enumerate(lines):
        if 'opencv_proj_mat =' in line:
            index = i + 1
    lines = lines[index: index + 3]
    lines = [list(map(float, line.split())) for line in lines]
    return np.array(lines)


def read_first_frame_pixels():
    return np.array([
        tuple(map(float, line.strip().split())) for line in open(first_frame_pixels_file).readlines()
    ])


def to_screen(point):
    return tuple(map(int, point))


def log_error_if_differ_much(diff, threshold, name):
    external_differ_output = name + ' differ by: ' + '%.4f' % diff
    if diff > threshold:
        logging.error(external_differ_output)
    else:
        print(external_differ_output)


def normalize_angle(angle):
    angle %= 2 * math.pi
    angle = min(angle, 2 * math.pi - angle)
    return angle


# Read input data
model_vertices = np.array(pywavefront.Wavefront(model_vertices_file).vertices)
camera_params = read_camera_params()
first_frame_pixels = read_first_frame_pixels()

# Find rotation and translation
_, rotation, translation = cv2.solvePnP(model_vertices, first_frame_pixels, camera_params, np.array([]))

# Find ground truth external parameters matrix
ground_truth_external_matrix = open(external_parameters_ground_truth).readlines()[0].split()
ground_truth_external_matrix = np.array(list(map(float, ground_truth_external_matrix))).reshape((4, 3)).T
ground_truth_external_matrix[:3, :3] = ground_truth_external_matrix[:3, :3].T

# Find external parameters matrix
R_rodrigues = cv2.Rodrigues(rotation)[0]
external_matrix = np.hstack((R_rodrigues, translation))
print("External matrix:\n", external_matrix)
print("External matrix ground truth:\n", ground_truth_external_matrix)


def screen_points(camera_params, external_matrix, model_vertices):
    projection_matrix = np.dot(camera_params, external_matrix)
    model_vertices_matrix = np.hstack((model_vertices, np.ones((len(model_vertices), 1)))).T
    points_homogeneous = np.dot(projection_matrix, model_vertices_matrix).T
    points = points_homogeneous[:, :2]
    points_w = points_homogeneous[:, 2:]
    points = points / points_w
    return points


def rotation_translation_points(matrix):
    rotation = matrix[:3, :3]
    translation = matrix[:, 3:4]
    points = screen_points(camera_params, external_matrix, model_vertices)
    return rotation, translation, points


def rotation_difference(actual, expected):
    diff_matrix = np.dot(actual, np.linalg.inv(expected))
    diff_direction, diff_angle = transforms3d.axangles.mat2axangle(diff_matrix)
    diff_angle = normalize_angle(diff_angle)
    log_error_if_differ_much(diff_angle, 0.04, 'External parameters rotation')
    return diff_angle


def translation_difference(actual, expected):
    diff = distance.euclidean(actual, expected)
    log_error_if_differ_much(diff, 5.0, 'External parameters translation')
    return diff


def points_difference(actual, expected):
    diff = 0
    for act, exp in zip(actual, expected):
        diff += distance.euclidean(act, exp)
    diff /= len(actual)
    log_error_if_differ_much(diff, 0.1, 'Projected points')
    return diff


def difference(actual, expected):
    actual_rotation, actual_translation, actual_points = rotation_translation_points(actual)
    expected_rotation, expected_translation, expected_points = rotation_translation_points(expected)
    rotation_difference(actual_rotation, expected_rotation)
    translation_difference(actual_translation, expected_translation)
    points_difference(actual_points, expected_points)


difference(external_matrix, ground_truth_external_matrix)

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
