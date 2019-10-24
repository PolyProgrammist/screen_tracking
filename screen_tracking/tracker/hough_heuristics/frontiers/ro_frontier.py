from screen_tracking.tracker.hough_heuristics.candidates import RoCandidate

from .frontier import Frontier


class RoFrontier(Frontier):
    def __init__(self, frontier, last_frame_line):
        super().__init__(frontier.tracker)
        self.candidates = [RoCandidate(last_frame_line, candidate, self.tracker_params) for candidate in
                           frontier.top_current()]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_RO
