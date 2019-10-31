from screen_tracking.tracker.hough_heuristics.candidates import RectangleShowKCandidate

from .frontier import Frontier


class RectangleShowKFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        self.max_show_count = 5
        top = frontier.top_current()
        self.candidates = [RectangleShowKCandidate(candidate) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PNP
