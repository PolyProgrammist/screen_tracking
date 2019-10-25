import cv2

from screen_tracking.tracker.hough_heuristics.candidates.phi_candidate import PhiCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_lines


class PhiFrontier(Frontier):
    def __init__(self, hough_frontier, last_frame_line):
        super().__init__()
        self.tracker_params = hough_frontier.tracker_params
        self.state = hough_frontier.state
        self.candidates = [PhiCandidate(last_frame_line, candidate) for candidate in hough_frontier.candidates]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PHI

    def show(self, need_to_show=False):
        show_lines(self.state.cur_frame.copy(), self.top_current(), need_to_show)
