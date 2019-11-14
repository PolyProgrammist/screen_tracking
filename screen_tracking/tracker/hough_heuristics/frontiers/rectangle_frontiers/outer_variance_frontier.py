from screen_tracking.tracker.hough_heuristics.candidates import OuterVarianceCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier


class OuterVarianceFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        self.candidates = [
            OuterVarianceCandidate(
                candidate,
                self.tracker.state.cur_frame,
                self.tracker_params.OUTER_VARIANCE_SHIFT_LINE
            )
            for candidate in top
        ]

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_OUTER_VARIANCE
