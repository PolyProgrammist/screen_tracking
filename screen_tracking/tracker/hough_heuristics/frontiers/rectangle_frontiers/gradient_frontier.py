from screen_tracking.tracker.hough_heuristics.candidates import RectangleGradientCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_frame
from screen_tracking.tracker.hough_heuristics.utils import processed_sobel


class RectangleGradientFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        frame = self.tracker.state.cur_frame
        frame = frame / 255
        sobel_x = processed_sobel(frame, 1, 0)
        sobel_y = processed_sobel(frame, 0, 1)
        # show_frame(sobel_x)
        self.candidates = [RectangleGradientCandidate(candidate, sobel_x, sobel_y) for candidate in top]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_GRADIENT_RECT
