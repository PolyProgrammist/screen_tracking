from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw


class InOutCandidate(Candidate):
    def __init__(self, inner, outer):
        super().__init__()
        self.inner = inner
        self.outer = outer

    def draw(self, frame):
        rectangle_draw(frame, self.inner, (0, 0, 255))
        rectangle_draw(frame, self.outer, (0, 255, 0))
