from screen_tracking.tracker.hough_heuristics.candidates.ground_truth_candidate import GroundTruthCandidate
from screen_tracking.tracker.hough_heuristics.candidates.previous_pos_candidate import PreviousPoseCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.common.utils import TrackingDataReader


class PreviousPoseFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        self.candidates = [PreviousPoseCandidate(candidate, self.tracker) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PREVIOUS_POSE
