from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate


class InOutCandidate(Candidate):
    def __init__(self, inner, outer):
        super().__init__()
        self.inner = inner
        self.outer = outer

    def draw(self, frame):
        rectangle_draw(frame, self.inner)
        rectangle_draw(frame, self.outer)
