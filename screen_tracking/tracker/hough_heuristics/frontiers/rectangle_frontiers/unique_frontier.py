import itertools

from screen_tracking.tracker.hough_heuristics.candidates import RectCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class RectUniqueFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        self.max_show_count = 1
        lines = []
        self.candidates = []
        for candidate in frontier.candidates:
            if candidate.lines not in lines:
                self.candidates.append(candidate)
                lines.append(candidate.lines)
