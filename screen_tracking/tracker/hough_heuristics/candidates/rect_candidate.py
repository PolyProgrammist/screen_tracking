from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate


class RectCandidate(Candidate):
    def __init__(self, line_candidates):
        super().__init__()
        self.current_score_ = 0
        self.overall_score_ = 0
        self.line_candidates = line_candidates
        self.tracker_params = line_candidates[0].tracker_params

    def show(self):
