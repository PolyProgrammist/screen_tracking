from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw, points_to_abc

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import distance_point_to_abc


class DistanceInOutCandidate(Candidate):
    def __init__(self, candidate):
        super().__init__()
        self.inner = candidate.inner
        self.outer = candidate.outer
        delta = 0
        for inner_line, outer_line in zip(self.inner.lines, self.outer.lines):
            inner_line = inner_line.line
            outer_line = outer_line.line
            delta += 1. / (self.distance(inner_line, outer_line) + 0.00001)
        self.current_score_ = delta
        # print(delta)

    def distance(self, inner_line, outer_line):
        point = (inner_line[0] + inner_line[1]) / 2
        return distance_point_to_abc(point, outer_line)

    def draw(self, frame):
        rectangle_draw(frame, self.inner)
        rectangle_draw(frame, self.outer)
