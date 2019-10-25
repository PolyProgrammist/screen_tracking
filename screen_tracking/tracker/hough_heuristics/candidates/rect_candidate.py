from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw

from .candidate import Candidate


class RectCandidate(Candidate):
    def __init__(self, lines):
        super().__init__()
        self.lines = lines

    def draw(self, frame):
        rectangle_draw(frame, self)
