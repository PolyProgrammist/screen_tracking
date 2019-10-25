import cv2


class Candidate:
    def __init__(self):
        self.overall_score_ = 0
        self.current_score_ = 0

    def current_score(self):
        return self.current_score_

    def overall_score(self):
        return self.overall_score_

    def draw(self, frame):
        pass


def draw_line(frame, line):
    cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), color=(0, 0, 255), thickness=1)
