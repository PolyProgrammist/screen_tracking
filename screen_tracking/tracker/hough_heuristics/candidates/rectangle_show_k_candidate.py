import numpy as np

from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw
from .candidate import Candidate


class RectangleShowKCandidate(Candidate):
    def __init__(self, candidate):
        super().__init__()
        self.lines = candidate.lines
        self.current_score_ = candidate.current_score_

    def draw(self, frame, color=None, only_lines=None):
        rectangle_draw(frame, self, color, only_lines)
