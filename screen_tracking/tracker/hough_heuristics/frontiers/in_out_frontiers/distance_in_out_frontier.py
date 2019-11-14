from screen_tracking.tracker.hough_heuristics.candidates import DistanceInOutCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class DistanceInOutFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        # TODO: max show count for each rectangle and in_out frontier. Consider making base class
        self.max_show_count = 1
        top = frontier.top_current()
        self.candidates = [DistanceInOutCandidate(candidate, self.tracker_params) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_IN_OUT_DISTANCE
