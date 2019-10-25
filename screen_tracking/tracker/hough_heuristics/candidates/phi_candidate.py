import numpy as np

from screen_tracking.common.common import normalize_angle
from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import draw_line
from screen_tracking.tracker.hough_heuristics.utils.geom2d import adjusted_abc


class PhiCandidate(Candidate):
    def __init__(self, last_frame_line, candidate):
        super().__init__()
        self.line = candidate.line
        self.current_score_ = self.collinear_predicate(last_frame_line, self.line)
        self.overall_score_ = self.current_score_
        self.scores = {}
        self.scores['phi'] = self.current_score_

    def direction_diff(self, a, b):
        phi_a = np.arctan2(a[0], a[1])
        phi_b = np.arctan2(b[0], b[1])
        return normalize_angle(phi_a - phi_b, round=np.math.pi)

    def collinear_predicate(self, last_frame_line, candidate_line):
        return self.direction_diff(adjusted_abc(last_frame_line), adjusted_abc(candidate_line))

    def draw(self, frame):
        draw_line(frame, self.line)
