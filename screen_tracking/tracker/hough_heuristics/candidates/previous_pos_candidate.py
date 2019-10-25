import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points
from screen_tracking.tracker.hough_heuristics.utils.geom3d import get_external_matrix, external_matrices_difference


class PreviousPoseCandidate(Candidate):
    def __init__(self, candidate, tracker):
        super().__init__()
        self.tracker = tracker
        self.lines = candidate.lines
        self.current_score_ = self.previous_matrix_diff([line.line for line in self.lines])

    def previous_matrix_diff(self, lines):
        intersections = screen_lines_to_points(lines)
        diff = external_matrices_difference(self.tracker, intersections)
        return diff[0] + diff[1]

    def draw(self, frame):
        rectangle_draw(frame, self)
