from screen_tracking.tracker.hough_heuristics.candidates import PreviousPoseCandidate

from .frontier import Frontier


class PreviousPoseFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        self.candidates = [PreviousPoseCandidate(candidate, self.tracker) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PREVIOUS_POSE
