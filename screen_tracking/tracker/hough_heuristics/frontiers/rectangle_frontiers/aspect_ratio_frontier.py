from screen_tracking.tracker.hough_heuristics.candidates.rectangle_candidates.aspect_ratio_candidate import \
    AspectRatioCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.tracker.hough_heuristics.utils import screen_points_to_lines


class AspectRatioFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        top = frontier.top_current()
        self.max_show_count = 1
        previous_aspect_ratio = self.previous_aspect_ratio()
        self.candidates = [AspectRatioCandidate(candidate, previous_aspect_ratio) for candidate in top]

    def previous_aspect_ratio(self):
        lines = screen_points_to_lines(self.state.last_points)
        return AspectRatioCandidate.aspect_ratio(lines)

    def max_diff_score(self):
        return self.tracker_params.MAX_DIFF_ASPECT_RATIO
