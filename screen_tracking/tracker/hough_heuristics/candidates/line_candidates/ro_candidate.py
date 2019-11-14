import numpy as np

from screen_tracking.tracker.hough_heuristics.utils import draw_line, adjusted_abc
from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate


# TODO: compare distance between lines (distance between their centers, for example) instead of rho
from screen_tracking.tracker.hough_heuristics.utils.geom2d import distance_lines_predicate


class PreviousLineDistanceCandidate(Candidate):
    def __init__(self, last_frame_line, candidate, tracker_params):
        super().__init__()
        self.tracker_params = tracker_params
        self.line = candidate.line
        self.current_score_ = distance_lines_predicate(last_frame_line, self.line)

    def draw(self, frame):
        draw_line(frame, self.line)
