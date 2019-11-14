from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import (
    rectangle_draw,
    screen_lines_to_points,
    difference_with_predicted
)


class PreviousPoseCandidate(Candidate):
    def __init__(self, candidate, tracker):
        super().__init__()
        self.tracker = tracker
        self.lines = candidate.lines
        self.current_score_ = self.previous_matrix_diff([line.line for line in self.lines])

    def previous_matrix_diff(self, lines):
        intersections = screen_lines_to_points(lines)
        diff = difference_with_predicted(self.tracker, intersections)
        # TODO: make more serious metric
        return diff[0] + diff[1]

    def draw(self, frame):
        rectangle_draw(frame, self)
