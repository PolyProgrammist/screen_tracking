import logging

from screen_tracking.common import TrackingDataReader
from screen_tracking.tracker.hough_heuristics.candidates.line_candidates.ground_truth_line_candidate import \
    GroundTruthLineCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.tracker.hough_heuristics.utils import get_screen_points, screen_points_to_lines


class GroundTruthLineFrontier(Frontier):
    def __init__(self, frontier, line_number):
        super().__init__(frontier.tracker)
        logging.error('Using Ground Truth Line Frontier')

        reader = TrackingDataReader()
        frame_number = self.tracker.state.frame_number
        ground_truth_matrix = reader.get_ground_truth()[frame_number]

        points = get_screen_points(self.tracker, ground_truth_matrix)
        lines = screen_points_to_lines(points)

        self.candidates = [
            GroundTruthLineCandidate(candidate, lines[line_number],
                                     self.tracker_params.MAX_DIFF_GROUND_TRUTH_LINE_DISTANCE)
            for candidate in frontier.top_current()
        ]
        self.candidates = self.top_current()[:1]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_GROUND_TRUTH_LINE_PHI
