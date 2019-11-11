from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import direction_diff


class PhiInOutCandidate(Candidate):
    def __init__(self, candidate):
        super().__init__()
        self.inner = candidate.inner
        self.outer = candidate.outer
        delta = 0
        for inner_line, outer_line in zip(self.inner.lines, self.outer.lines):
            inner_line = inner_line.line
            outer_line = outer_line.line
            delta += direction_diff(inner_line, outer_line)
        self.current_score_ = delta

    def draw(self, frame):
        rectangle_draw(frame, self.inner)
        rectangle_draw(frame, self.outer)
