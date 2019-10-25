import cv2

from screen_tracking.tracker.hough_heuristics.utils import screen_lines_to_points, screen_points_to_lines


class Frontier:
    def __init__(self):
        self.candidates = []

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

    def show(self):
        pass

    def max_diff_score(self):
        pass


def show_lines(frontier, **kwargs):
    cur_frame = kwargs.get('frame', frontier.state.cur_frame.copy())
    for candidate in frontier.top_current(**kwargs):
        candidate.draw(cur_frame)
    if not kwargs.get('no_show'):
        cv2.imshow('Phi frontier', cur_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# def show(candidate):
#     points = screen_lines_to_points(candidate.line_candidates)
#     lines = screen_points_to_lines(points)
#
#     to_show = candidate.tracker_param
#     for line in lines:
#         line = line.astype(int)
#         cv2.line(to_show, tuple(line[0]), tuple(line[1]), color=(0, 0, 255), thickness=1)
#     for point in points:
#         cv2.circle(to_show, tuple(point.astype(int)), 5, color=(0, 255, 0), thickness=-1)
#     cv2.imshow('frame', to_show)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# def show_rectangle(candidate):
#     pass
