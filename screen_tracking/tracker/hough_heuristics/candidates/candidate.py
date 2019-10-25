import cv2

from screen_tracking.tracker.hough_heuristics.utils import screen_lines_to_points, screen_points_to_lines, draw_line


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
