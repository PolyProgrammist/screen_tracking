from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import rectangle_draw


class RectCandidate(Candidate):
    def __init__(self, lines):
        super().__init__()
        self.lines = lines

    def draw(self, frame):
        rectangle_draw(frame, self)
