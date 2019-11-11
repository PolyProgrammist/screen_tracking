from screen_tracking.tracker.hough_heuristics.candidates import LineGradientCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.tracker.hough_heuristics.utils import processed_sobel


class LineGradientFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        # TODO: same in RectangleGradientFrontier, duplicate code
        frame = self.tracker.state.cur_frame
        frame = frame / 255
        sobelx = processed_sobel(frame, 1, 0)
        sobely = processed_sobel(frame, 0, 1)
        self.candidates = [LineGradientCandidate(candidate, sobelx, sobely) for candidate in frontier.top_current()]

    def max_diff_score(self):
        return self.tracker_params.MAX_GRADIENT
