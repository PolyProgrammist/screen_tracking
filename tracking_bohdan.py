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
        verbose=2
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


def main():
    img_0 = (cv2.imread(imgname(1), cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)
    pose_0 = np.array([
        [1, 0, -105.0],
        [0, 1, -340.0]
    ])

    for img_number in range(2, 100):
        img_1 = (cv2.imread(imgname(img_number), cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)

        # show_diff(img_0, img_1, pose_0, pose_0)

        patch_size = [101, 101]
        pose_1 = track(patch_size, img_0, img_1, pose_0)

        if img_number % 10 == 0:
            real_img = cv2.imread(imgname(img_number))
            show_rectangle(real_img, patch_size, pose_1)

        img_0 = img_1
        pose_0 = pose_1

main()
