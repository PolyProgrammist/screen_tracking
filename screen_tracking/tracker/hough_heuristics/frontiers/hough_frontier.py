import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates import HoughCandidate
from screen_tracking.tracker.hough_heuristics.utils import cut, get_bounding_box

from .frontier import Frontier


class HoughFrontier(Frontier):
    def __init__(self, tracker):
        super().__init__(tracker)
        hough_lines = self.hough_lines(
            self.state.cur_frame,
            get_bounding_box(
                self.state.cur_frame.shape,
                self.state.last_points,
                self.tracker_params.MARGIN_FRACTION)
        )
        self.candidates = [HoughCandidate(line) for line in hough_lines]

    def hough_lines(self, cur_frame_init, bounding_box):
        cur_frame = cut(cur_frame_init, bounding_box)
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            self.tracker_params.CANNY_THRESHOLD_1,
            self.tracker_params.CANNY_THRESHOLD_2,
            apertureSize=self.tracker_params.APERTURE_SIZE,
            L2gradient=True
        )
        cv2.imshow('edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        lines = cv2.HoughLinesP(
            edges,
            1,
            1 * np.pi / 180,
            self.tracker_params.THRESHOLD_HOUGH_LINES_P,
            minLineLength=self.tracker_params.MIN_LINE_LENGTH,
            maxLineGap=self.tracker_params.MAX_LINE_GAP
        )
        lines = [line[0].astype(float) for line in lines]
        lines = [
            line + np.array([bounding_box[0], bounding_box[1], bounding_box[0], bounding_box[1]]) for line in lines
        ]
        lines = [[line[:2], line[2:]] for line in lines]
        return np.array(lines)
