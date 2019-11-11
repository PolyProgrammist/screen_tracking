from screen_tracking.tracker.hough_heuristics.candidates import PNPrmseCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class PNPrmseFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        self.max_show_count = 1
        top = frontier.top_current()
        self.candidates = [PNPrmseCandidate(candidate, frontier.tracker) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PNP
