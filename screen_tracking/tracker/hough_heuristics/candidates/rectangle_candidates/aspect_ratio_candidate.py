import numpy as np
from cv2 import cv2

from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw,
    screen_lines_to_points,
    mean_gradient)

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import polyarea, screen_points_to_lines, intersected_lines


class RectangleGradientCandidate(Candidate):
    def __init__(self, candidate):
        super().__init__()
        self.lines = candidate.lines
        self.current_score_ = self.aspect_ratio([line.line for line in self.lines])

    def aspect_ratio(self, lines):
        return 0

    def draw(self, frame):
        rectangle_draw(frame, self)
