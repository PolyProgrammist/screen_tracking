import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points
from screen_tracking.tracker.hough_heuristics.utils.geom3d import get_external_matrix, get_screen_points


class PNPrmseCandidate(Candidate):
    def __init__(self, candidate, tracker):
        super().__init__()
        self.tracker = tracker
        self.lines = candidate.lines
        self.current_score_ = self.pnp_rmse([line.line for line in self.lines])

    def pnp_rmse(self, lines):
        intersections = screen_lines_to_points(lines)
        external_matrix = get_external_matrix(self.tracker, intersections)
        points = get_screen_points(self.tracker, external_matrix)
        return np.linalg.norm(points - intersections)

    def draw(self, frame):
        rectangle_draw(frame, self)
