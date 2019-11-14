import transforms3d
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates import SquareCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class SquareFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1

        # TODO: square limit according to view angle
        # print(self.tracker.predict_matrix)
        # extrinsic = self.tracker.state.predict_matrix
        # rotation = extrinsic[:3, :3]
        # diff_direction, diff_angle = transforms3d.axangles.mat2axangle(rotation)
        # print(diff_angle, np.cos(diff_angle))

        self.candidates = [SquareCandidate(candidate, self.tracker) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_SQUARE
