from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points, \
    my_segment_distance


class DistanceInOutCandidate(Candidate):
    def __init__(self, candidate, tracker_params):
        super().__init__()
        self.inner = candidate.inner
        self.outer = candidate.outer
        delta = 0
        for inner_line, outer_line in zip(self.inner.lines, self.outer.lines):
            inner_line = inner_line.line
            outer_line = outer_line.line
            distance = my_segment_distance(inner_line, outer_line)
            # TODO: change magic number or change 1 / x to -x
            delta += 1. / (distance + 0.00001)
            if distance < tracker_params.SMALLEST_SCREEN_FRAME or distance > tracker_params.LARGEST_SCREEN_FRAME:
                self.allowed_ = False
        self.current_score_ = delta
        points_inner = screen_lines_to_points([candidate.line for candidate in self.inner.lines])
        points_outer = screen_lines_to_points([candidate.line for candidate in self.outer.lines])
        # TODO: move this logic to another frontier
        # TODO: assure inner is inner truly (not just x1 < x2 etc.)
        x = 0
        y = 1
        bottom_left = 0
        bottom_right = 1
        top_right = 2
        top_left = 3
        if points_inner[bottom_left][x] < points_outer[bottom_left][x] or \
                points_inner[bottom_left][y] > points_outer[bottom_left][y] or \
                points_inner[bottom_right][x] > points_outer[bottom_right][x] or \
                points_inner[bottom_right][y] > points_outer[bottom_right][y] or \
                points_inner[top_left][x] < points_outer[top_left][x] or \
                points_inner[top_left][y] < points_outer[top_left][y] or \
                points_inner[top_right][x] > points_outer[top_right][x] or \
                points_inner[top_right][y] < points_outer[top_right][y]:
            self.allowed_ = False

    def draw(self, frame):
        rectangle_draw(frame, self.inner)
        rectangle_draw(frame, self.outer)
