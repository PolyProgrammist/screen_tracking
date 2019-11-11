import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import draw_line, mean_gradient


class LineGradientCandidate(Candidate):
    def __init__(self, candidate, sobelx, sobely):
        super().__init__()
        self.line = candidate.line
        mean_grad = mean_gradient(self.line, sobelx, sobely)
        # TODO: change 1 / x to -x
        self.current_score_ = 1 / mean_grad if mean_grad else np.inf

    def draw(self, frame):
        draw_line(frame, self.line)
