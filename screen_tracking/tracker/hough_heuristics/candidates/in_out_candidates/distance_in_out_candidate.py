from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw, points_to_abc

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import distance_point_to_abc, screen_lines_to_points


class DistanceInOutCandidate(Candidate):
    def __init__(self, candidate, tracker_params):
        super().__init__()
        self.inner = candidate.inner
        self.outer = candidate.outer
        delta = 0
        for inner_line, outer_line in zip(self.inner.lines, self.outer.lines):
            inner_line = inner_line.line
            outer_line = outer_line.line
            distance = self.distance(inner_line, outer_line)
            delta += 1. / (distance + 0.00001)
            if distance < tracker_params.SMALLEST_SCREEN_FRAME or distance > tracker_params.LARGEST_SCREEN_FRAME:
                self.allowed_ = False
        self.current_score_ = delta
        points_inner = screen_lines_to_points([candidate.line for candidate in self.inner.lines])
        points_outer = screen_lines_to_points([candidate.line for candidate in self.outer.lines])
        x = 0
        y = 1
        bottom_left = 0
        bottom_right = 1
        top_right = 2
        top_left = 3
        if points_inner[bottom_left][x] < points_outer[bottom_left][x] or\
            points_inner[bottom_right][x] > points_outer[bottom_right][x] or\
            points_inner[bottom_left][y] > points_outer[bottom_left][y] or\
            points_inner[top_left][y] < points_outer[top_left][y]:
            self.allowed_ = False

    def distance(self, inner_line, outer_line):
        point = (inner_line[0] + inner_line[1]) / 2
        return distance_point_to_abc(point, outer_line)

    def draw(self, frame):
        rectangle_draw(frame, self.inner)
        rectangle_draw(frame, self.outer)
