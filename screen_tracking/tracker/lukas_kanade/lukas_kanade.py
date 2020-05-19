import logging

import numpy as np
import cv2
from scipy.optimize import least_squares

from screen_tracking.tracker.common_tracker import Tracker
from screen_tracking.tracker.hough_heuristics.utils import get_screen_points, screen_points_to_lines, draw_line
from screen_tracking.tracker.hough_heuristics.utils.draw import show_frame
from screen_tracking.tracker.lukas_kanade.lukas_params import TrackerParams


#! /usr/bin/env python3

import cv2
import numpy as np
from scipy.optimize import least_squares


def to_mat(vec):
    a, sx, sy, m, tx, ty = vec
    r_mat = np.array([
        [np.cos(a), -np.sin(a)],
        [np.sin(a), np.cos(a)]
    ])
    shear_mat = np.array([
        [1.0, m],
        [0.0, 1.0]
    ])
    scale_mat = np.array([
        [sx, 0],
        [0, sy]
    ])
    mat = np.zeros((2, 3))
    mat[:, :2] = r_mat @ shear_mat @ scale_mat
    mat[:, 2] = np.array([tx, ty])
    return mat

    # rotation + translation
    # return np.array([
    #     [np.cos(a), -np.sin(a), vec[1]],
    #     [np.sin(a), np.cos(a), vec[2]]
    # ])


def from_mat(mat):
    tx, ty = mat[:, 2]
    a = mat[:, :2]
    sx = (a[0, 0] ** 2 + a[1, 0] ** 2) ** 0.5
    angle = np.arctan(a[1, 0] / a[0, 0])
    msy = a[0, 1] * np.cos(angle) + a[1, 1] * np.sin(angle)
    if np.isclose(angle, 0):
        sy = (a[1, 1] - msy * np.sin(angle)) / np.cos(angle)
    else:
        sy = (msy * np.cos(angle) - a[0, 1]) / np.sin(angle)
    m = msy / sy
    return np.array([angle, sx, sy, m, tx, ty])

    # rotation + translation
    # return np.array([np.arcsin(mat[1, 0]), mat[0, 2], mat[1, 2]])


def track(patch_size, img_0, img_1, pose_0):
    assert(img_0.shape == img_1.shape)
    img_size = img_0.shape[1], img_1.shape[0]

    img_0_warped = cv2.warpAffine(img_0, pose_0, img_size)

    rs, cs = patch_size

    def func(x):
        h = to_mat(x + from_mat(pose_0))
        img_1_warped = cv2.warpAffine(img_1, h, img_size)
        residuals = (img_1_warped[:rs, :cs] - img_0_warped[:rs, :cs]).flatten()
        # print((residuals ** 2).sum())
        return residuals

    x0 = np.zeros((6,))
    result = least_squares(
        func,
        x0,
        diff_step=np.array([0.0001, 0.001, 0.001, 0.001, 0.1, 0.1]),
        jac='3-point',
        xtol=1e-3,
        gtol=1e-3,
        ftol=1e-3,
        # verbose=2
    )

    return to_mat(result.x + from_mat(pose_0))


def show_diff(img_0, img_1, pose_0, pose_1):
    # cv2.imshow('0', img_0)
    # cv2.imshow('1', img_1)
    img_0_warped = img_0
    img_1_warped = img_1
    img_0_warped = cv2.warpAffine(img_0, pose_0, (img_0.shape[1], img_0.shape[0]))
    img_1_warped = cv2.warpAffine(img_1, pose_1, (img_1.shape[1], img_1.shape[0]))
    img_diff = np.abs(img_0_warped - img_1_warped)
    cv2.imshow('diff', img_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imgname(number):
    number = str(number)
    number = '0' * (4 - len(number)) + number
    result = 'card/' + number + '.jpg'
    return result


def inverse_pose(y, P):
    A = P[:, :2]
    t = P[:, 2:]
    y = np.array(y).reshape(-1, 1)
    x = np.linalg.inv(A) @ (y - t)
    return x.reshape(-1)


def show_rectangle(img, patch_size, pose):
    points = [
        [0, 0],
        [0, patch_size[1]],
        [patch_size[0], patch_size[1]],
        [patch_size[0], 0]
    ]
    result = []
    for point in points:
        result.append(inverse_pose(point, pose))

    np.array(result, np.int32)
    img = img.copy()
    cv2.polylines(img, np.array([result], np.int32), True, (0, 255, 255))
    cv2.imshow('rect', img)
    cv2.waitKey(0)

def convert_to_track(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    return img.astype(np.float32)

def bohdan_tracking(cur_frame, last_frame, last_points):
    img_0 = convert_to_track(last_frame)
    img_1 = convert_to_track(cur_frame)
    patch_size = [101, 101]

    ret_points = []
    for i in range(4):
        point = last_points[i]
        translation = -(point - np.array(patch_size) / 2.0)
        pose_0 = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]]
        ])
        pose_1 = track(patch_size, img_0, img_1, pose_0)
        to_invert = np.array(patch_size) / 2
        point = inverse_pose(to_invert, pose_1)
        ret_points.append(point)

    ret_points = np.array(ret_points)
    # print(ret_points)
    return ret_points



class LukasKanadeTracker(Tracker):
    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        super().__init__(model_vertices, camera_params, video_source, frame_pixels)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        return bohdan_tracking(cur_frame, last_frame, last_points)
