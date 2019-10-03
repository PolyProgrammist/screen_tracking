import math
import numpy as np


def screen_points(camera_params, external_matrix, model_vertices):
    projection_matrix = np.dot(camera_params, external_matrix)
    model_vertices_matrix = np.hstack((model_vertices, np.ones((len(model_vertices), 1)))).T
    points_homogeneous = np.dot(projection_matrix, model_vertices_matrix).T
    points = points_homogeneous[:, :2]
    points_w = points_homogeneous[:, 2:]
    points = points / points_w
    return points


def normalize_angle(angle):
    angle %= 2 * math.pi
    angle = min(angle, 2 * math.pi - angle)
    return angle
