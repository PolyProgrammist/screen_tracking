import cv2


class Frontier:
    def __init__(self):
        self.candidates = []

    def top_overall(self, count=None):
        result = list(sorted(self.candidates, key=lambda candidate: candidate.overall_score_))
        return result if count is None else result[:count]

    def top_current(self, count=None):
        result = list(sorted(self.candidates, key=lambda candidate: candidate.current_score_))
        result = [t for t in result if t.current_score_ <= self.max_diff_score()]
        return result if count is None else result[:count]

    def overall_current(self):
        return min(self.candidates, key=lambda candidate: candidate.overall_score_)

    def best_current(self):
        return min(self.candidates, key=lambda candidate: candidate.current_score_)

    def show(self):
        pass

    def max_diff_score(self):
        pass


def show_lines(cur_frame, candidates, need_to_show):
    for candidate in candidates:
        candidate.draw(cur_frame)
    if need_to_show:
        cv2.imshow('Phi frontier', cur_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
