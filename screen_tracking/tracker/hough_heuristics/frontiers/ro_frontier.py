from screen_tracking.tracker.hough_heuristics.candidates.ro_candidate import RoCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_best


class RoFrontier(Frontier):
    def __init__(self, frontier, last_frame_line, **top):
        super().__init__(frontier.tracker)
        self.candidates = [RoCandidate(last_frame_line, candidate, self.tracker_params) for candidate in
                           frontier.top_current(**top)]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_RO
