import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import rectangle_draw
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points
from screen_tracking.tracker.hough_heuristics.utils.geom3d import get_external_matrix, external_matrices_difference


class PreviousPoseCandidate(Candidate):
    def __init__(self, candidate, tracker):
        super().__init__()
        self.lines = candidate.lines
        self.tracker = tracker
        self.diff = self.previous_matrix_diff([line.line for line in self.lines])
        self.current_score_ = self.diff[0] + self.diff[1]
        self.scores = candidate.scores
        self.scores['previous_pose_angle'] = self.diff[0]
        self.scores['previous_pose_translation'] = self.diff[1]

        phi = sum(line['phi'] for line in self.scores['lines'])
        ro = sum(line['ro'] for line in self.scores['lines'])
        pnprmse = self.scores['pnprmse']
        angle = self.scores['previous_pose_angle']
        translation = self.scores['previous_pose_translation']

        # self.current_score_ = phi * 2000 + ro + 4 * pnprmse + 100 * angle + 100 * translation
        # print(self.current_score_)

    def previous_matrix_diff(self, lines):
        intersections = screen_lines_to_points(lines)
        diff = external_matrices_difference(self.tracker, intersections)
        return diff

    def draw(self, frame):
        rectangle_draw(frame, self)
