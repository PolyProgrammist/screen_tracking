from screen_tracking.tracker.hough_heuristics.candidates import PhiCandidate

from .frontier import Frontier


class PhiFrontier(Frontier):
    def __init__(self, frontier, last_frame_line):
        super().__init__(frontier.tracker)
        self.candidates = [PhiCandidate(last_frame_line, candidate) for candidate in frontier.top_current()]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PHI
