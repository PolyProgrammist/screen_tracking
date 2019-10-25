import logging

from screen_tracking.tracker.hough_heuristics.candidates.ground_truth_candidate import GroundTruthCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.common.utils import TrackingDataReader


class GroundTruthFrontier(Frontier):
    def __init__(self, frontier):
        logging.error('Using Ground Truth Frontier')
        super().__init__(frontier.tracker)
        self.max_show_count = 1

        reader = TrackingDataReader()
        frame_number = 2
        ground_truth_matrix = reader.get_ground_truth()[frame_number]

        top = frontier.top_current()
        self.candidates = [GroundTruthCandidate(candidate, ground_truth_matrix, frontier.tracker) for candidate in top]

    def max_diff_score(self):
        return 5
