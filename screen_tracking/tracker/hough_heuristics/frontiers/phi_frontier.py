import cv2

from screen_tracking.tracker.hough_heuristics.candidates.phi_candidate import PhiCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_best


class PhiFrontier(Frontier):
    def __init__(self, frontier, last_frame_line):
        super().__init__(frontier.tracker)
        self.candidates = [PhiCandidate(last_frame_line, candidate) for candidate in frontier.top_current()]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_PHI
