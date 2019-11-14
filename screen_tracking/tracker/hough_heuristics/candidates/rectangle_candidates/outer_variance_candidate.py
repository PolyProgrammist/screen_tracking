import numpy as np
from cv2 import cv2

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import show_frame
from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw,
    screen_lines_to_points,
    mean_gradient)

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import polyarea, screen_points_to_lines, perpendicular, \
    intersected_lines
from screen_tracking.tracker.hough_heuristics.utils.img_utils import image_line_index


class OuterVarianceCandidate(Candidate):
    def __init__(self, candidate, frame, outer_variance_shift):
        super().__init__()
        self.lines = candidate.lines
        self.outer_variance_shift = outer_variance_shift
        self.current_score_ = self.variance([line.line for line in self.lines], frame)

    def variance(self, lines, frame):
        lines = intersected_lines(lines)
        var = []
        for i, line in enumerate(lines):
            theta = perpendicular(line)
            next_point = lines[(i + 1) % len(lines)][0]
            inner = next_point - line[1]
            if np.dot(theta, inner) > 0:
                theta = - theta
            line += theta * self.outer_variance_shift
            var = np.append(var, image_line_index(line, frame))

        result = np.var(var, axis=0).mean()
        return result

    def draw(self, frame):
        rectangle_draw(frame, self)
