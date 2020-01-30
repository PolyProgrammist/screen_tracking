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


def track(patch_size, img_0, img_1, pose_0, pose_1=None, verbose=0):
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

    if pose_1 is None:
        pose_1 = pose_0
    x0 = from_mat(pose_1) - from_mat(pose_0)
    result = least_squares(
        func,
        x0,
        diff_step=np.array([0.0001, 0.001, 0.001, 0.001, 0.1, 0.1]),
        jac='3-point',
        xtol=1e-3,
        gtol=1e-3,
        ftol=1e-3,
        verbose=verbose
    )

    return to_mat(result.x + from_mat(pose_0))


def show_diff(img_0, img_1, pose_0, pose_1):
    img_0_warped = cv2.warpAffine(img_0, pose_0, (img_0.shape[1], img_0.shape[0]))
    img_1_warped = cv2.warpAffine(img_1, pose_1, (img_1.shape[1], img_1.shape[0]))
    img_diff = np.abs(img_0_warped - img_1_warped)
    cv2.imshow('Diff', img_diff)
    cv2.waitKey(0)


def downscale_pose(pose_mat, lvl):
    new_pose = pose_mat.copy()
    new_pose[:, 2] /= 2 ** lvl
    return new_pose


def track_pyr(patch_size, img_0, img_1, pose_0, lvl_count=3, verbose=0):
    pyramid_0 = [img_0]
    pyramid_1 = [img_1]
    for lvl in range(lvl_count - 1):
        pyramid_0.append(cv2.pyrDown(pyramid_0[-1]))
        pyramid_1.append(cv2.pyrDown(pyramid_1[-1]))

    pose = downscale_pose(pose_0, lvl_count - 1)
    for lvl in range(lvl_count - 1, -1, -1):
        pose_opt = track(
            patch_size,
            pyramid_0[lvl],
            pyramid_1[lvl],
            downscale_pose(pose_0, lvl),
            pose,
            verbose
        )
        pose = pose_opt
        if lvl > 0:
            pose[:, 2] *= 2
    return pose


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
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)
    return img

def convert_to_track(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    return img.astype(np.float32)


def main():
    # img_0 = (cv2.imread(imgname(1), cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)
    # img_0 = cv2.imread(imgname(1))
    # img_0 = convert_to_track(img_0)

    cap = cv2.VideoCapture('sequence.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = 'out.mov'
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))



    ret, frame = cap.read()
    img_0 = convert_to_track(frame)
    # cv2.imwrite('1.jpg', frame)
    # cv2.imshow('img', frame)
    # cv2.waitKey(0)
    pose_0 = np.array([
        # [1, 0, -105.0],
        # [0, 1, -340.0]
        [1, 0, -200.0],
        [0, 1, -720.0]
    ])

    img_number = 0
    while cap.isOpened():
        print(img_number)
        ret, frame = cap.read()
        if not ret:
            break
        img_number += 1
        img_1 = convert_to_track(frame)
        # img_1 = (cv2.imread(imgname(img_number), cv2.IMREAD_GRAYSCALE) / 255.0).astype(np.float32)

        # show_diff(img_0, img_1, pose_0, pose_0)

        patch_size = [101, 101]
        pose_1 = track(patch_size, img_0, img_1, pose_0)

        # if img_number % 10 == 0:
        real_img = frame
        show_rectangle(real_img, patch_size, pose_1)

        img_0 = img_1
        pose_0 = pose_1

        print(np.max(img_1), np.min(img_1))

        img_to_write = (img_1 * 255).astype(np.uint8)
        print(np.min(img_to_write), np.max(img_to_write))
        if ret:
            # cv2.imshow('img', img_to_write)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            out.write(show_rectangle(frame, patch_size, pose_1))

    out.release()

main()