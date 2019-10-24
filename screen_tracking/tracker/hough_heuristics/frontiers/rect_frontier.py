import itertools

from screen_tracking.tracker.hough_heuristics.candidates import RectCandidate

from .frontier import Frontier


class RectFrontier(Frontier):
    def __init__(self, side_frontiers):
        super().__init__(side_frontiers[0].tracker)
        self.max_show_count = 1
        sides = [side.top_current() for side in side_frontiers]
        self.candidates = [RectCandidate(list(t)) for t in itertools.product(*sides)]
