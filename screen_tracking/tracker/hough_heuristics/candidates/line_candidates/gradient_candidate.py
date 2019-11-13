import numpy as np

from screen_tracking.tracker.hough_heuristics.utils import draw_line

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import direction_diff
from skimage import draw


class GradientCandidate(Candidate):
    def __init__(self, candidate, sobelx, sobely):
        super().__init__()
        self.line = candidate.line
        line = np.hstack((self.line[:, 1:], self.line[:, :1]))  # swap x and y
        deltax = line[0][0] - line[1][0]
        deltay = line[0][1] - line[1][1]
        theta = np.array([-deltay, deltax])
        discrete_line = draw.line(
            *np.round(line[0]).astype(np.int),
            *np.round(line[1]).astype(np.int)
        )
        dx = sobelx[discrete_line].reshape(-1, 1)
        dy = sobely[discrete_line].reshape(-1, 1)
        gradients = np.hstack((dx, dy))
        oriented_gradients = gradients * theta
        result = np.mean(np.abs(np.sum(oriented_gradients, axis=1)))
        self.current_score_ = 1. / result if result else np.inf

    def draw(self, frame):
        draw_line(frame, self.line)
