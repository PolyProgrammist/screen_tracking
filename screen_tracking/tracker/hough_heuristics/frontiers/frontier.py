import cv2


class Frontier:
    def __init__(self):
        self.candidates = []

    def top_overall(self, count=None):
        result = list(sorted(self.candidates, key=lambda candidate: candidate.overall_score_))
        return result if count is None else result[:count]

    def top_current(self, **kwargs):
        result = list(sorted(self.candidates, key=lambda candidate: candidate.current_score_))
        if kwargs.get('all_candidates'):
            return result
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
