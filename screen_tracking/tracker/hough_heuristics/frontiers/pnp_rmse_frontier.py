from screen_tracking.tracker.hough_heuristics.candidates.pnp_rmse_candidate import PNPrmseCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class PNPrmseFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        self.candidates = [PNPrmseCandidate(candidate, frontier.tracker) for candidate in top]

    def max_diff_score(self):
        return 5
