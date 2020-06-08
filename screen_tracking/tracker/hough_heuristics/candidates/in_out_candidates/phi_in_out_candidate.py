from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.candidates.in_out_candidates.in_out_candidate import InOutCandidate
from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import direction_diff


class PhiInOutCandidate(InOutCandidate):
    def __init__(self, candidate):
        super().__init__(candidate.inner, candidate.outer)
        delta = 0
        for i, inner_line, outer_line in zip(range(len(self.inner.lines)), self.inner.lines, self.outer.lines):
            inner_line = inner_line.line
            outer_line = outer_line.line
            dir_diff = direction_diff(inner_line, outer_line)
            # TODO: handle outer frame differently, make it's own model
            if i != 0:
                delta += dir_diff
        self.current_score_ = delta
