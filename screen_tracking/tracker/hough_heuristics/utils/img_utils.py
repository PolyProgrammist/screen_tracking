import numpy as np
from skimage import draw


def mean_gradient(line, sobelx, sobely):
    line = np.hstack((line[:, 1:], line[:, :1]))  # swap x and y
    deltax = line[0][0] - line[1][0]
    deltay = line[0][1] - line[1][1]
    theta = np.array([-deltay, deltax], dtype=np.float)
    theta /= np.linalg.norm(theta)
    point_a = np.round(line[0]).astype(np.int)
    point_b = np.round(line[1]).astype(np.int)
    discrete_line = draw.line(*point_a, *point_b)
    dx = sobelx[discrete_line].reshape(-1, 1)
    dy = sobely[discrete_line].reshape(-1, 1)
    gradients = np.hstack((dx, dy))
    oriented_gradients = gradients * theta
    result = np.mean(np.abs(np.sum(oriented_gradients, axis=1)))
    return result
