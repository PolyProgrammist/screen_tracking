import numpy as np

from screen_tracking.tracker.hough_heuristics.utils import draw_line, adjusted_abc
from .candidate import Candidate


class RoCandidate(Candidate):
    def __init__(self, last_frame_line, candidate, tracker_params):
        super().__init__()
        self.tracker_params = tracker_params
        self.line = candidate.line
        self.current_score_ = self.distance_origin_near_predicate(last_frame_line, self.line)

    def distance_to_origin_diff(self, a, b):
        return np.abs(np.abs(a[2]) - np.abs(b[2]))

    def distance_origin_near_predicate(self, last_frame_line, candidate_line):
        return self.distance_to_origin_diff(adjusted_abc(last_frame_line), adjusted_abc(candidate_line))

    def draw(self, frame):
        draw_line(frame, self.line)
