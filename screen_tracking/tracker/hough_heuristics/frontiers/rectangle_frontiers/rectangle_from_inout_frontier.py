from screen_tracking.tracker.hough_heuristics.candidates import RectCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class RectFromInOutFrontier(Frontier):
    def __init__(self, in_out_frontier):
        super().__init__(in_out_frontier.tracker)
        self.max_show_count = 1
        self.candidates = [RectCandidate(candidate.inner.lines) for candidate in in_out_frontier.top_current()]
