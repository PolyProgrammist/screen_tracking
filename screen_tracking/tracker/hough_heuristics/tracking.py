import cv2
import numpy as np


class Tracker:
    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels

    def get_bounding_box(self, cur_frame, last_points):
        margin_fraction = 1 / 5

        length = 0
        for i in range(4):
            for j in range(4):
                cur_length = np.linalg.norm(last_points[i] - last_points[j])
                length = max(length, cur_length)
        offset_len = length * margin_fraction
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

    def adjust_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def points_to_abc(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        ar = np.array([a, b, c])
        return ar

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
        canny_threshold1 = 50
        canny_threshold2 = 150
        minLineLength = 100
        maxLineGap = 30
        threshold_hough_lines_p = 100

        cur_frame = self.cut(cur_frame_init, bbox)
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold_hough_lines_p, minLineLength=minLineLength,
                                maxLineGap=maxLineGap)
        lines = [line[0].astype(float) for line in lines]
        lines = [line + np.array([bbox[0], bbox[1], bbox[0], bbox[1]]) for line in lines]
        lines = [[line[:2], line[2:]] for line in lines]
        return np.array(lines)

    def near_lines(self, a, b):
        return min(np.linalg.norm(a - b), np.linalg.norm(-a - b))

    def collinear_predicate(self, line, candidate):
        line_abc = self.adjust_vector(self.points_to_abc(line[0], line[1]))
        candidate_abc = self.adjust_vector(self.points_to_abc(candidate[0], candidate[1]))
        value = self.near_lines(line_abc, candidate_abc)
        return value

    def filter_lines(self, last_lines, candidates, predicate):
        return np.array(sorted(candidates.tolist(), key=lambda candidate: predicate(last_lines, candidate))[:1])

    def get_only_lines(self, line, candidates):
        return candidates[0]

    def show(self, lines, frame, points):
        to_show = frame.copy()
        for line in lines:
            line = line.astype(int)
            cv2.line(to_show, tuple(line[0]), tuple(line[1]), color=(0, 0, 255), thickness=2)
        for point in points:
            cv2.circle(to_show, tuple(point.astype(int)), 5, color=(0, 255, 0), thickness=-1)
        cv2.imshow('frame', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_points(self, cur_frame, last_frame, last_points):
        last_lines = self.screen_points_to_lines(last_points)
        hough_lines = self.hough_lines(cur_frame, self.get_bounding_box(cur_frame, last_points))
        candidates = [hough_lines.copy(), hough_lines.copy(), hough_lines.copy(), hough_lines.copy()]
        result = []
        for line, candidate in zip(last_lines, candidates):
            candidate = self.filter_lines(line, candidate, self.collinear_predicate)
            result.append(self.get_only_lines(line, candidate))
        intersections = self.screen_lines_to_points(result)
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
        self.write_camera(tracking_result, last_points[0], 1)
        frame_number = 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            points = self.get_points(frame, last_frame[0], last_points[0])
            self.write_camera(tracking_result, points, frame_number)
            last_points[0] = points
            last_frame[0] = frame
            frame_number += 1

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file):
    result = Tracker(model_vertices, camera_params, video_source, frame_pixels).track()
    write_callback(result)
