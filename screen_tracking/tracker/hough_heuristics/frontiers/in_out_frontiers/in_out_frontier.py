import itertools

from screen_tracking.tracker.hough_heuristics.candidates import InOutCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class InOutFrontier(Frontier):
    def __init__(self, rect_frontier):
        super().__init__(rect_frontier.tracker)
        self.max_show_count = 1
        top = rect_frontier.top_current()
        self.candidates = [InOutCandidate(inner, outer) for (inner, outer) in itertools.permutations(top, 2)]
