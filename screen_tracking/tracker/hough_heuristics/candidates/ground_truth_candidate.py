from screen_tracking.test.compare import difference as external_matrices_difference
from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points
from screen_tracking.tracker.hough_heuristics.utils.geom3d import get_external_matrix


class GroundTruthCandidate(Candidate):
    def __init__(self, candidate, ground_truth_matrix, tracker):
        super().__init__()
        self.lines = candidate.lines
        self.tracker = tracker
        self.ground_truth_matrix = ground_truth_matrix
        self.current_score_ = self.ground_truth_difference([line.line for line in self.lines])

    def ground_truth_difference(self, lines):
        intersections = screen_lines_to_points(lines)
        return external_matrices_difference(
            get_external_matrix(self.tracker, intersections),
            self.ground_truth_matrix,
            self.tracker.camera_params,
            self.tracker.model_vertices
        )[2]

    def draw(self, frame):
        rectangle_draw(frame, self)
