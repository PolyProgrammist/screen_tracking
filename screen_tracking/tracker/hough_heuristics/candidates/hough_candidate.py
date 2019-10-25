import cv2

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate
from screen_tracking.tracker.hough_heuristics.utils.draw import draw_line


class HoughCandidate(Candidate):
    def __init__(self, line):
        super().__init__()
        self.line = line

    def draw(self, frame):
        draw_line(frame, self.line)
