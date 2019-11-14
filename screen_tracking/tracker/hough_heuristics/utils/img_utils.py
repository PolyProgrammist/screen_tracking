import numpy as np
from skimage import draw

from screen_tracking.tracker.hough_heuristics.utils.geom2d import perpendicular


def mean_gradient(line, sobelx, sobely):
    theta = perpendicular(line)
    # TODO: use image_line_index
    line = np.hstack((line[:, 1:], line[:, :1]))  # swap x and y
    point_a = np.round(line[0]).astype(np.int)
    point_b = np.round(line[1]).astype(np.int)
    discrete_line = draw.line(*point_a, *point_b)
    dx = sobelx[discrete_line].reshape(-1, 1)
    dy = sobely[discrete_line].reshape(-1, 1)
    gradients = np.hstack((dx, dy))
    oriented_gradients = gradients * theta
    result = np.mean(np.abs(np.sum(oriented_gradients, axis=1)))
    return result


def image_line_index(line, image):
    line = np.hstack((line[:, 1:], line[:, :1]))  # swap x and y
    point_a = np.round(line[0]).astype(np.int)
    point_b = np.round(line[1]).astype(np.int)
    discrete_line = draw.line(*point_a, *point_b)
    return image[discrete_line]
