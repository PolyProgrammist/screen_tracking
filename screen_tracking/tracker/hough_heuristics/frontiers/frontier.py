import cv2

from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points, screen_points_to_lines


class Frontier:
    def __init__(self, tracker):
        self.candidates = []
        self.tracker = tracker
        self.tracker_params = self.tracker.tracker_params
        self.state = self.tracker.state
        self.max_show_count = None

    def top_current(self, **kwargs):
        comparator = (lambda candidate: candidate.overall_score_) if kwargs.get('overall_score') else (lambda candidate: candidate.current_score_)
        result = list(sorted(self.candidates, key=comparator))
        if not kwargs.get('any_value'):
            result = [t for t in result if t.current_score_ <= self.max_diff_score()]
        return result[:kwargs['max_count']] if 'max_count' in kwargs else result

    def overall_current(self):
        return min(self.candidates, key=lambda candidate: candidate.overall_score_)

    def best_current(self):
        return min(self.candidates, key=lambda candidate: candidate.current_score_)

    def max_diff_score(self):
        return 1


def show_best(frontier, **kwargs):
    cur_frame = kwargs.get('frame', frontier.state.cur_frame.copy())
    top = frontier.top_current(**kwargs)
    if frontier.max_show_count:
        top = top[:frontier.max_show_count]
    for candidate in top:
        candidate.draw(cur_frame)
    if not kwargs.get('no_show'):
        cv2.imshow('Frontier', cur_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
