from screen_tracking.tracker.hough_heuristics.candidates.ro_candidate import RoCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_lines


class RoFrontier(Frontier):
    def __init__(self, phi_frontier, last_frame_line):
        super().__init__()
        self.tracker_params = phi_frontier.tracker_params
        self.state = phi_frontier.state
        self.candidates = [RoCandidate(last_frame_line, candidate, self.tracker_params) for candidate in
                           phi_frontier.candidates]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_RO

    def show(self, need_to_show=False):
        show_lines(self.state.cur_frame.copy(), self.top_current(), need_to_show)
