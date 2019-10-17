import cv2

from screen_tracking.common.utils import TrackingDataReader
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

import cv2
import numpy as np

from screen_tracking.common.utils import TrackingDataReader


class Tracker:
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
        offset_len = length / 5
        dxs = [-1, 0, 1, 0]
        dys = [0, 1, 0, -1]
        offsets = [[offset_len * dx, offset_len * dy] for dx, dy in zip(dxs, dys)]
        points = [point + offset for offset in offsets for point in last_points]

        min_x = np.clip(min(p[0] for p in points), 0, cur_frame.shape[1])
        max_x = np.clip(max(p[0] for p in points), 0, cur_frame.shape[1])
        min_y = np.clip(min(p[1] for p in points), 0, cur_frame.shape[0])
        max_y = np.clip(max(p[1] for p in points), 0, cur_frame.shape[0])

        result = min_x, max_x, min_y, max_y

        result = tuple(map(int, result))

        return result

    def cut(self, frame, bbox):
        min_x, max_x, min_y, max_y = bbox
        return frame[min_y:max_y, min_x:max_x, :]

    def adjust_vector(self, vector):
        result = vector.copy()
        result /= np.linalg.norm(result)
        if result[0] < 0:
            result *= -1
        return result

    def points_to_abc(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        ar = np.array([a, b, c])
        ar = self.adjust_vector(ar)
        return ar

    def screen_points_to_lines(self, last_points):
        lines = []
        for i in range(4):
            lines.append(self.points_to_abc(last_points[i], last_points[(i + 1) % 4]))
        return lines

    def hough_lines(self, cur_frame):
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minLineLength = 100
        maxLineGap = 30
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = [line[0].astype(float) for line in lines]
        lines_abc = [self.points_to_abc(line[:2], line[2:4]) for line in lines]
        return lines_abc, lines

    def near_lines(self, a, b):
        return np.linalg.norm(a - b)

    def filter_lines(self, last_lines, candidates_abc, candidates):
        result = []
        for candidate_abc, candidate in zip(candidates_abc, candidates):
            min_near = min(self.near_lines(last_line, candidate_abc) for last_line in last_lines)
            result.append((min_near, candidate_abc, candidate))
        result = sorted(result, key=lambda x: x[0])
        result = result[:20]
        nearest_abc = [r[1] for r in result]
        nearest = [r[2] for r in result]
        return nearest_abc, nearest

    def show(self, lines, frame):
        to_show = frame.copy()
        for line in lines:
            line = line.astype(int)
            cv2.line(to_show, (line[0], line[1]), (line[2], line[3]), color=(0, 0, 255))
        cv2.imshow('frame', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_points(self, cur_frame, last_frame, last_points):
        bbox = self.get_bounding_box(cur_frame, last_points)
        bbox = (0, cur_frame.shape[1], 0, cur_frame.shape[0])

        cur_frame = self.cut(cur_frame, bbox)
        last_frame = self.cut(last_frame, bbox)
        last_lines = self.screen_points_to_lines(last_points)
        hough_lines_abc, hough_lines = self.hough_lines(cur_frame)
        line_candidates_abc, line_candidates = self.filter_lines(last_lines, hough_lines_abc, hough_lines)
        self.show(line_candidates, last_frame)
        print(len(line_candidates))
        print(len(hough_lines))
        cur_abc = best_paralellogram(cur_abc)
        cur_points = points_from_lines(cur_abc)
        return cur_points

    def write_camera(self, tracking_result, pixels, frame):
        _, rotation, translation = cv2.solvePnP(self.model_vertices, pixels, self.camera_params, np.array([]))
        R_rodrigues = cv2.Rodrigues(rotation)[0]
        external_matrix = np.hstack((R_rodrigues, translation))
        tracking_result[frame] = external_matrix

    def track(self):
        tracking_result = {}

        cap = cv2.VideoCapture(self.video_source)

        ret, last_frame = cap.read()
        last_points = self.frame_pixels[1]
        self.write_camera(tracking_result, last_points, 1)
        frame_number = 2

        while cap.isOpened():
            ret, frame = cap.read()
            points = self.get_points(frame, last_frame, last_points)
            self.write_camera(tracking_result, points, frame_number)
            last_points = points
            last_frame = frame
            frame_number += 1
            break

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file):
    result = Tracker(model_vertices, camera_params, video_source, frame_pixels).track()
    write_callback(result)


if __name__ == "__main__":
    pass
    # reader = TrackingDataReader()
    # track(*reader.tracker_input())

# reader = TrackingDataReader('resources/tests/generated_tv_on', test_description='test_description.yml')
# video_file = reader.relative_file(reader.get_sequence_file())
# cap = cv2.VideoCapture(video_file)


#
#
# def draw_lines(img, houghLines, thickness=2):
#     color = [0, 0, 255]
#     for line in houghLines:
#         for rho, theta in line:
#             cv2.circle(img, rho, theta, 3, color, thickness)
#
#
# def go_frame(cnt):
#     ret, frame = cap.read()
#     img = frame
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     cv2.imshow('edges', edges)
#     minLineLength = 100
#     maxLineGap = 30
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
#     tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
#     h, theta, d = hough_line(edges, theta=tested_angles)
#     # img = np.zeros()
#
#     hough = np.log(1 + h)
#     print(hough.shape)
#     print(np.max(hough))
#     hough /= np.max(hough)
#     hough = hough * 255
#     hough = hough.astype(np.uint8)
#
#     hough_add = np.zeros(hough.shape, dtype=np.uint8)
#     hough_add = cv2.cvtColor(hough_add, cv2.COLOR_GRAY2BGR)
#     for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.2 * np.max(h))):
#         print(angle, _)
#         y = _
#         x = angle * 360 / np.math.pi
#         cv2.circle(hough_add, (int(x), int(y)), 3, [0, 255, 0], 3)
#
#     hough = cv2.cvtColor(hough, cv2.COLOR_GRAY2BGR)
#     cv2.imshow('hough', hough)
#
#     hough = cv2.resize(hough, (500, 500))
#
#     cv2.imwrite('resources/tmp/hough' + str(cnt) + '.jpg', hough)
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#     cv2.imwrite('resources/tmp/detected' + str(cnt) + '.jpg', img)
#
#
# go_frame(1)
# go_frame(2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
