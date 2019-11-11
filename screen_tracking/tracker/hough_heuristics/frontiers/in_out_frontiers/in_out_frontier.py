import itertools

from screen_tracking.tracker.hough_heuristics.candidates import InOutCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class InOutFrontier(Frontier):
    def __init__(self, rect_frontier_in, rect_frontier_out):
        super().__init__(rect_frontier_in.tracker)
        self.max_show_count = 1
        top_in = rect_frontier_in.top_current()
        top_out = rect_frontier_out.top_current()
        self.candidates = [InOutCandidate(inner, outer) for (inner, outer) in itertools.product(top_in, top_out)]
