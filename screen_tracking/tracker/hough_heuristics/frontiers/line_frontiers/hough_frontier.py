import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.candidates import HoughCandidate
from screen_tracking.tracker.hough_heuristics.utils import cut, get_bounding_box, screen_points_to_lines, \
    lines_intersection

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
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            self.tracker_params.CANNY_THRESHOLD_1,
            self.tracker_params.CANNY_THRESHOLD_2,
            apertureSize=self.tracker_params.APERTURE_SIZE,
            L2gradient=self.tracker_params.L2_GRADIENT
        )

        # show_frame(edges)
        # lines = cv2.HoughLines(edges, 1, 0.5 * np.pi / 180, 150)
        #
        # thetas = []
        #
        # x_mx = cur_frame.shape[1]
        # y_mx = cur_frame.shape[0]
        # margin = 3.0
        # first = np.array([margin, y_mx - margin])
        # second = np.array([x_mx - margin, y_mx - margin])
        # third = np.array([x_mx - margin, margin])
        # fourth = np.array([margin, margin])
        # points_boundary = [first, second, third, fourth]
        # lines_boundary = screen_points_to_lines(points_boundary)
        #
        # def intersect_frame(x1, y1, x2, y2):
        #     line = np.array([[x1, y1], [x2, y2]])
        #     for i, bound in enumerate(lines_boundary):
        #         try:
        #             point = lines_intersection(line, bound)
        #         except:
        #             continue
        #
        #         if point[0] > 0 and point[0] < x_mx and point[1] > 0 and point[1] < y_mx:
        #             cur_vec = line[0] - line[1]
        #             next_point = lines_boundary[(i + 1) % len(lines_boundary)][1]
        #             inner = next_point - bound[1]
        #             if np.dot(cur_vec, inner) < 0:
        #                 line[0] = point
        #             else:
        #                 line[1] = point
        #
        #     line = line.astype(np.int)
        #
        #     return line[0][0], line[0][1], line[1][0], line[1][1]
        #
        # for line in lines:
        #     for rho, theta in line:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))
        #         x1, y1, x2, y2 = intersect_frame(x1, y1, x2, y2)
        #         thetas.append(theta)
        #         cur_frame = cv2.line(cur_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #
        # show_frame(cur_frame)
        #
        lines = cv2.HoughLinesP(
            edges,
            self.tracker_params.HOUGH_DISTANCE_RESOLUTION * 1,
            self.tracker_params.HOUGH_ANGLE_RESOLUTION * np.pi / 180,
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
