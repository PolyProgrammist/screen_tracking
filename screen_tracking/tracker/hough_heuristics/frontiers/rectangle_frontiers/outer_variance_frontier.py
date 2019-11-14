import numpy as np
from screen_tracking.tracker.hough_heuristics.candidates import RectangleGradientCandidate, OuterVarianceCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_frame
from screen_tracking.tracker.hough_heuristics.utils import processed_sobel


class OuterVarianceFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        frame = self.tracker.state.cur_frame.copy()
        self.candidates = [
            OuterVarianceCandidate(candidate, self.tracker.state.cur_frame, self.tracker_params.OUTER_VARIANCE_SHIFT)
            for candidate in top
        ]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_GRADIENT_RECT
