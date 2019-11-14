from screen_tracking.tracker.hough_heuristics.candidates import PreviousLineDistanceCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class PreviousLineDistanceFrontier(Frontier):
    def __init__(self, frontier, last_frame_line, inner=True):
        super().__init__(frontier.tracker)
        self.candidates = [PreviousLineDistanceCandidate(last_frame_line, candidate, self.tracker_params) for candidate in
                           frontier.top_current()]
        self.max_diff = self.tracker_params.MAX_DIFF_LINE_DISTANCE_INNER \
            if inner else self.tracker_params.MAX_DIFF_LINE_DISTANCE_OUTER

    def max_diff_score(self):
        return self.max_diff
