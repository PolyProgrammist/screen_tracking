import cv2

from screen_tracking.tracker.hough_heuristics.utils import screen_lines_to_points, screen_points_to_lines


class Frontier:
    def __init__(self, tracker):
        self.candidates = []
        self.tracker = tracker
        self.tracker_params = self.tracker.tracker_params
        self.state = self.tracker.state
        self.max_show_count = None

    def top_current(self, **kwargs):
        comparator = (lambda candidate: candidate.current_score_)
        result = list(sorted(self.filter(self.candidates), key=comparator))
        if kwargs.get('starting_point'):
            result = result[kwargs['starting_point']:]
        if not kwargs.get('any_value'):
            result = [t for t in result if t.current_score_ <= self.max_diff_score()]
        return result[:kwargs['max_count']] if 'max_count' in kwargs else result

    def filter(self, candidates):
        return [candidate for candidate in candidates if candidate.allowed_]

    def max_diff_score(self):
        return 1

    def print_best(self, caption=None, print_top_scores=False):
        if not caption:
            caption = self.__class__.__name__
        top = self.top_current()
        print(caption, 'len = ', len(top))
        if print_top_scores:
            print([candidate.current_score_ for candidate in top[:print_top_scores]])


def show_frame(cur_frame, caption='Frontier'):
    cv2.imshow(caption, cur_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_best(frontier, **kwargs):
    top = frontier.top_current(**kwargs)
    if kwargs.get('show_all'):
        for i, candidate in enumerate(top):
            print('Candidate number: ', i + 1 + kwargs.get('starting_point', 0))
            cur_frame = frontier.state.cur_frame.copy()
            candidate.draw(cur_frame)
            show_frame(cur_frame)
    else:
        if frontier.max_show_count:
            top = top[:frontier.max_show_count]
        cur_frame = kwargs.get('frame', frontier.state.cur_frame.copy())
        for candidate in top:
            candidate.draw(cur_frame)
        if not kwargs.get('no_show'):
            show_frame(cur_frame)
