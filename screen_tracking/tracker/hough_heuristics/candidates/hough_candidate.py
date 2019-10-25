import cv2

from screen_tracking.tracker.hough_heuristics.candidates.candidate import Candidate


class HoughCandidate(Candidate):
    def __init__(self, line):
        super().__init__()
        self.line = line

    def draw(self, frame):
        cv2.line(frame, tuple(map(int, self.line[0])), tuple(map(int, self.line[1])), color=(0, 0, 255), thickness=1)
