from screen_tracking.tracker.hough_heuristics.candidates import SquareCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class SquareFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        self.candidates = [SquareCandidate(candidate, self.tracker) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_SQUARE
