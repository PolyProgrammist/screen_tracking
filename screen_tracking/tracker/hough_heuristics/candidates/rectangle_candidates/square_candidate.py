import numpy as np
from cv2 import cv2

from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw,
    screen_lines_to_points,
    get_external_matrix,
    difference_with_predicted
)

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import polyarea


class SquareCandidate(Candidate):
    def __init__(self, candidate, tracker):
        super().__init__()
        self.tracker = tracker
        self.lines = candidate.lines
        self.current_score_ = self.area_change([line.line for line in self.lines])

    def area_change(self, lines):
        intersections = screen_lines_to_points(lines)
        previous = self.tracker.state.last_points
        previous_area = polyarea(previous)
        current_area = polyarea(intersections)
        result = np.abs((previous_area - current_area) / previous_area)
        return result

    def draw(self, frame):
        rectangle_draw(frame, self)
