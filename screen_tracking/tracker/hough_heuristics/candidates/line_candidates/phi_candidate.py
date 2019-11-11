import numpy as np

from screen_tracking.common import normalize_angle
from screen_tracking.tracker.hough_heuristics.utils import draw_line, adjusted_abc

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate


class PhiCandidate(Candidate):
    def __init__(self, last_frame_line, candidate):
        super().__init__()
        self.line = candidate.line
        self.current_score_ = self.collinear_predicate(last_frame_line, self.line)

    @staticmethod
    def direction_diff(a, b):
        phi_a = np.arctan2(a[0], a[1])
        phi_b = np.arctan2(b[0], b[1])
        return normalize_angle(phi_a - phi_b, round=np.math.pi)

    def collinear_predicate(self, last_frame_line, candidate_line):
        return self.direction_diff(adjusted_abc(last_frame_line), adjusted_abc(candidate_line))

    def draw(self, frame):
        draw_line(frame, self.line)
