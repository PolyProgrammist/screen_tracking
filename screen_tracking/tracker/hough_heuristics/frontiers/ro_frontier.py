from screen_tracking.tracker.hough_heuristics.candidates.ro_candidate import RoCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_lines


class RoFrontier(Frontier):
    def __init__(self, frontier, last_frame_line):
        super().__init__()
        self.tracker_params = frontier.tracker_params
        self.state = frontier.state
        self.candidates = [RoCandidate(last_frame_line, candidate, self.tracker_params) for candidate in
                           frontier.top_current()]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_RO

    def show(self, **kwargs):
        show_lines(self, **kwargs)
