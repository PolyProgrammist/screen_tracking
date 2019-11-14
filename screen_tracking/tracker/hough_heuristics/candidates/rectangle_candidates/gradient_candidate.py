import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw,
    mean_gradient)
from screen_tracking.tracker.hough_heuristics.utils.geom2d import intersected_lines


class RectangleGradientCandidate(Candidate):
    def __init__(self, candidate, sobelx, sobely):
        super().__init__()
        self.lines = candidate.lines
        self.current_score_ = self.gradient([line.line for line in self.lines], sobelx, sobely)

    def gradient(self, lines, sobelx, sobely):
        lines = intersected_lines(lines)
        result = 0
        lines = lines
        for line in lines:
            result += mean_gradient(line, sobelx, sobely)
        result /= len(lines)
        return 1 / result if result else np.inf

    def draw(self, frame):
        rectangle_draw(frame, self)
