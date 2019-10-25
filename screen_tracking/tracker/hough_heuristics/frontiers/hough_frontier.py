import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates.hough_candidate import HoughCandidate
from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier
from screen_tracking.tracker.hough_heuristics.utils import cut, get_bounding_box


class HoughFrontier(Frontier):
    def __init__(self, tracker):
        super().__init__()
        self.tracker_params = tracker.tracker_params
        self.state = tracker.state
        hough_lines = self.hough_lines(self.state.cur_frame,
                                       get_bounding_box(self.state.cur_frame, self.state.last_points,
                                                        self.tracker_params.MARGIN_FRACTION))
        self.candidates = [HoughCandidate(line) for line in hough_lines]

    def hough_lines(self, cur_frame_init, bbox):
        cur_frame = cut(cur_frame_init, bbox)
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.tracker_params.CANNY_THRESHOLD_1, self.tracker_params.CANNY_THRESHOLD_2,
                          apertureSize=self.tracker_params.APERTURE_SIZE, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.tracker_params.THRESHOLD_HOUGH_LINES_P,
                                minLineLength=self.tracker_params.MIN_LINE_LENGTH,
                                maxLineGap=self.tracker_params.MAX_LINE_GAP)
        lines = [line[0].astype(float) for line in lines]
        lines = [line + np.array([bbox[0], bbox[1], bbox[0], bbox[1]]) for line in lines]
        lines = [[line[:2], line[2:]] for line in lines]
        return np.array(lines)

    def show(self):
        cur_frame = self.state.cur_frame.copy()
        for candidate in self.candidates:
            candidate.draw(cur_frame)
        cv2.imshow('Hough lines', cur_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def max_diff_score(self):
        return 1