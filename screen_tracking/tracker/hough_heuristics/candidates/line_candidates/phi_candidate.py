import numpy as np

from screen_tracking.tracker.hough_heuristics.utils import draw_line

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.geom2d import direction_diff


class PhiCandidate(Candidate):
    def __init__(self, last_frame_line, candidate):
        super().__init__()
        self.line = candidate.line
        self.current_score_ = direction_diff(last_frame_line, self.line)

    def draw(self, frame):
        draw_line(frame, self.line)
