import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates import HoughCandidate
from screen_tracking.tracker.hough_heuristics.utils import cut, get_bounding_box

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import Frontier, show_frame


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

        # ddepth = cv2.CV_16S
        # kernel_size = 3
        # src = cv2.GaussianBlur(cur_frame, (3, 3), 0)
        # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
        # abs_dst = cv2.convertScaleAbs(dst)
        # print(abs_dst.dtype)
        # show_frame(abs_dst)
        # ret, frame = cv2.threshold(abs_dst, 20, 255, cv2.THRESH_BINARY)
        # show_frame(frame)
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            self.tracker_params.CANNY_THRESHOLD_1,
            self.tracker_params.CANNY_THRESHOLD_2,
            apertureSize=self.tracker_params.APERTURE_SIZE,
            L2gradient=self.tracker_params.L2_GRADIENT
        )
        show_frame(edges)
        lines = cv2.HoughLinesP(
            edges,
            self.tracker_params.HOUGH_DISTANCE_RESOLUTION * 1,
            self.tracker_params.HOUGH_ANGLE_RESOLUTION * np.pi / 180,
            self.tracker_params.THRESHOLD_HOUGH_LINES_P,
            minLineLength=self.tracker_params.MIN_LINE_LENGTH,
            maxLineGap=self.tracker_params.MAX_LINE_GAP
        )
        lines = [line[0].astype(float) for line in lines]
        # hlines = cv2.HoughLines(edges, 1, 3 * np.pi / 180, 30)
        # lines = []
        # print(hlines.shape)
        # for line in hlines:
        #     rho, theta = line[0][0], line[0][1]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #     lines.append([x1, y1, x2, y2])
        # print(hlines)
        # print(lines)

        lines = [
            line + np.array([bounding_box[0], bounding_box[1], bounding_box[0], bounding_box[1]]) for line in lines
        ]
        lines = [[line[:2], line[2:]] for line in lines]
        return np.array(lines)
