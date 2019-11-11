from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import draw_line
from screen_tracking.tracker.hough_heuristics.utils.geom2d import direction_diff, distance_lines_predicate


class GroundTruthLineCandidate(Candidate):
    def __init__(self, candidate, ground_truth_line, max_line_distance):
        super().__init__()
        self.line = candidate.line
        self.current_score_ = direction_diff(ground_truth_line, self.line)
        self.allowed_ = distance_lines_predicate(ground_truth_line, self.line) < max_line_distance

    def draw(self, frame):
        draw_line(frame, self.line)
