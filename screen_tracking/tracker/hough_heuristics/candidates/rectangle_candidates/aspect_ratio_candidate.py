import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw)
from screen_tracking.tracker.hough_heuristics.utils.geom2d import intersected_lines, \
    my_segment_distance, aspect_ratio


class AspectRatioCandidate(Candidate):
    def __init__(self, candidate, previous_aspect_ratio):
        super().__init__()
        self.lines = candidate.lines
        self.current_score_ = self.aspect_ratio_diff([line.line for line in self.lines], previous_aspect_ratio)

    def aspect_ratio_diff(self, lines, previous_aspect_ratio):
        return np.abs(self.aspect_ratio(lines) - previous_aspect_ratio) / previous_aspect_ratio

    @staticmethod
    def aspect_ratio(lines):
        # TODO: inherit intersected lines from parent, don't count each time
        lines = intersected_lines(lines)
        return aspect_ratio(lines)

    def draw(self, frame):
        rectangle_draw(frame, self)
