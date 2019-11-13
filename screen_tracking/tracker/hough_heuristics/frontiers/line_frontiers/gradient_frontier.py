import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates import GradientCandidate

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_frame


class GradientFrontier(Frontier):
    def __init__(self, frontier):
        super().__init__(frontier.tracker)
        frame = self.tracker.state.cur_frame
        frame = frame / 255
        sobelx = self.sobel(frame, 1, 0)
        sobely = self.sobel(frame, 0, 1)
        # show_frame(sobely)
        self.candidates = [GradientCandidate(candidate, sobelx, sobely) for candidate in frontier.top_current()]

    def sobel(self, frame, x, y):
        ksize = 3
        frame = cv2.Sobel(frame, cv2.CV_64F, x, y, ksize=ksize)
        frame = np.sum(frame, axis=2)
        # print(np.min())

        # sobel = cv2.morphologyEx(sobel, cv2.MORPH_OPEN, kernel=np.array([
        #     [1, 1, 1],
        #     [1, 1, 1],
        #     [1, 1, 1]
        # ]))

        ret, sobel_positive = cv2.threshold(frame, np.max(frame) * 0.05, np.max(frame), cv2.THRESH_TOZERO)
        frame = -frame
        ret, sobel_negative = cv2.threshold(frame, np.max(frame) * 0.05, np.max(frame), cv2.THRESH_TOZERO)
        sobel_negative = -sobel_negative
        frame = sobel_negative + sobel_positive

        return frame

    def max_diff_score(self):
        return self.tracker_params.MAX_GRADIENT
