from screen_tracking.tracker.hough_heuristics.candidates import PhiInOutCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class PhiInOutFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        self.max_show_count = 1
        top = frontier.top_current()
        self.candidates = [PhiInOutCandidate(candidate) for candidate in top]

    def max_diff_score(self):
        return 0.001
