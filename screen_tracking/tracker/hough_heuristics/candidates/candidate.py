import cv2

from screen_tracking.tracker.hough_heuristics.utils.draw import draw_line
from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points, screen_points_to_lines


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
