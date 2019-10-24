import math

import cv2
import numpy as np

from screen_tracking.common.common import normalize_angle, screen_points


class TrackerParams:
    MARGIN_FRACTION = 1 / 5
    CANNY_THRESHOLD_1 = 50
    CANNY_THRESHOLD_2 = 150
    MIN_LINE_LENGTH = 100
    MAX_LINE_GAP = 30
    THRESHOLD_HOUGH_LINES_P = 50
    APERTURE_SIZE = 3
    PNP_FLAG = cv2.SOLVEPNP_IPPE


class Tracker:
    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels

    def get_bounding_box(self, cur_frame, last_points):
        length = 0
        for i in range(4):
            for j in range(4):
                cur_length = np.linalg.norm(last_points[i] - last_points[j])
                length = max(length, cur_length)
        offset_len = length * self.tracker_params.MARGIN_FRACTION
        dxs = [-1, 0, 1, 0]
        dys = [0, 1, 0, -1]
        offsets = [[offset_len * dx, offset_len * dy] for dx, dy in zip(dxs, dys)]
        points = [point + offset for offset in offsets for point in last_points]

        min_x = np.clip(min(p[0] for p in points), 0, cur_frame.shape[1])
        max_x = np.clip(max(p[0] for p in points), 0, cur_frame.shape[1])
        min_y = np.clip(min(p[1] for p in points), 0, cur_frame.shape[0])
        max_y = np.clip(max(p[1] for p in points), 0, cur_frame.shape[0])

        result = min_x, min_y, max_x, max_y

        result = tuple(map(int, result))

        return result

    def cut(self, frame, bbox):
        min_x, min_y, max_x, max_y = bbox
        return frame.copy()[min_y:max_y, min_x:max_x, :]

    def adjust_vector_abc(self, vector):
        return vector / np.linalg.norm(vector[:2])

    def points_to_abc(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        ar = np.array([a, b, c])
        return ar

    def adjusted_abc(self, line):
        return self.adjust_vector_abc(self.points_to_abc(line[0], line[1]))

    def screen_lines_to_points(self, lines):
        intersections = []
        for i in range(-1, 3):
            intersections.append(self.lines_intersection(lines[(i + 4) % 4], lines[(i + 1) % 4]))
        return np.array(intersections)

    def screen_points_to_lines(self, last_points):
        lines = []
        for i in range(4):
            lines.append([last_points[i], last_points[(i + 1) % 4]])
        return np.array(lines)

    def lines_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return np.array([x, y])

    def hough_lines(self, cur_frame_init, bbox):
        cur_frame = self.cut(cur_frame_init, bbox)
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

    def direction_diff(self, a, b):
        phi_a = np.arctan2(a[0], a[1])
        phi_b = np.arctan2(b[0], b[1])
        return normalize_angle(phi_a - phi_b, round=math.pi)

    def distance_to_origin_diff(self, a, b):
        return np.abs(np.abs(a[2]) - np.abs(b[2]))

    def collinear_predicate(self, line, candidate):
        return self.direction_diff(self.adjusted_abc(line), self.adjusted_abc(candidate))

    def distance_origin_near_predicate(self, line, candidate):
        return self.distance_to_origin_diff(self.adjusted_abc(line), self.adjusted_abc(candidate))

    def combine_predicate(self, line, candidate):
        line = self.adjusted_abc(line)
        candidate = self.adjusted_abc(candidate)
        coeff = 400
        return self.direction_diff(line, candidate) * coeff + self.distance_to_origin_diff(line, candidate)

    def filter_lines(self, last_lines, candidates, predicate, max_diff, max_number=20):
        indices = list(range(len(candidates)))
        values = np.array([predicate(last_lines, candidates[index]) for index in indices])
        indices = sorted(indices, key=lambda x: values[x])
        a = indices[:max_number]
        b = np.nonzero(values < max_diff)[0]
        c = [t for t in a if t in b]
        return candidates[c]

    def show(self, lines, frame, points):
        lines = lines.copy()
        to_show = frame.copy()
        for line in lines:
            line = line.astype(int)
            cv2.line(to_show, tuple(line[0]), tuple(line[1]), color=(0, 0, 255), thickness=1)
        for point in points:
            cv2.circle(to_show, tuple(point.astype(int)), 5, color=(0, 255, 0), thickness=-1)
        cv2.imshow('frame', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pnp_rmse(self, lines):
        intersections = self.screen_lines_to_points(lines)
        _, rotation, translation = cv2.solvePnP(self.model_vertices, intersections, self.camera_params,
                                                np.array([]))
        R_rodrigues = cv2.Rodrigues(rotation)[0]
        external_matrix = np.hstack((R_rodrigues, translation))
        points = screen_points(self.camera_params, external_matrix, self.model_vertices)
        return np.linalg.norm(points - intersections)

    def best_rectangle(self, candidates):
        best_score = -1
        best_lines = []
        for bottom in candidates[0]:
            for right in candidates[1]:
                for top in candidates[2]:
                    for left in candidates[3]:
                        current = np.array([bottom, right, top, left])
                        rmse = self.pnp_rmse(current)
                        if best_score == -1 or rmse < best_score:
                            best_score = rmse
                            best_lines = current
        return best_lines

    def get_points(self, cur_frame, last_frame, last_points):
        last_lines = self.screen_points_to_lines(last_points)
        hough_lines = self.hough_lines(cur_frame, self.get_bounding_box(cur_frame, last_points))
        candidates = [hough_lines.copy(), hough_lines.copy(), hough_lines.copy(), hough_lines.copy()]
        result = []
        # candidates[3] = self.filter_lines(last_lines[3], candidates[3], self.collinear_predicate, max_diff=0.3)
        # result = self.filter_lines(last_lines[3], candidates[3], self.distance_origin_near_predicate, max_diff=100)
        for _, (line, candidate) in enumerate(zip(last_lines, candidates)):
            candidate = self.filter_lines(line, candidate, self.distance_origin_near_predicate, max_diff=30)
            candidate = self.filter_lines(line, candidate, self.collinear_predicate, max_diff=0.1)
            candidate = self.filter_lines(line, candidate, self.combine_predicate, max_diff=100000, max_number=10000)[
                        :10]
            result.append(candidate)

        resulttmp = result.copy()
        result = self.best_rectangle(result)

        intersections = self.screen_lines_to_points(result)
        # intersections = []
        # result = result[1:1]
        # self.show(result, cur_frame, intersections)

        return intersections

    def write_camera(self, tracking_result, pixels, frame):
        _, rotation, translation = cv2.solvePnP(self.model_vertices, pixels, self.camera_params, np.array([]))
        R_rodrigues = cv2.Rodrigues(rotation)[0]
        external_matrix = np.hstack((R_rodrigues, translation))
        tracking_result[frame] = external_matrix

    def track(self):
        tracking_result = {}

        cap = cv2.VideoCapture(self.video_source)

        ret, last_frame = cap.read()
        last_frame = [last_frame]
        last_points = [self.frame_pixels[1]]
        frame_number = 1
        self.write_camera(tracking_result, last_points[0], frame_number)
        frame_number = 2
        # return tracking_result

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            points = self.get_points(frame, last_frame[0], last_points[0])
            self.write_camera(tracking_result, points, frame_number)
            last_points[0] = points
            last_frame[0] = frame
            frame_number += 1
            print(frame_number)
            if frame_number == 84:
                break

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file, tracker_params=None):
    tracker = Tracker(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
