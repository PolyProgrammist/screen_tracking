import math
import itertools

import cv2
import numpy as np

from screen_tracking.common.common import normalize_angle, screen_points
from screen_tracking.test.compare import difference as external_matrices_difference
from screen_tracking.tracker.hough_heuristics.frontiers.hough_frontier import HoughFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.phi_frontier import PhiFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.ro_frontier import RoFrontier
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.utils import adjusted_abc, screen_lines_to_points, screen_points_to_lines, \
    get_bounding_box, cut


class Tracker:
    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels
        self.state = TrackerState()

    # def combine_predicate(self, line, candidate):
    #     line = adjusted_abc(line)
    #     candidate = adjusted_abc(candidate)
    #     coeff = 400
    #     return self.direction_diff(line, candidate) * coeff + self.distance_to_origin_diff(line, candidate)

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

    def get_external_matrix(self, points):
        _, rotation, translation = cv2.solvePnP(self.model_vertices, points, self.camera_params,
                                                np.array([]), flags=self.tracker_params.PNP_FLAG)
        R_rodrigues = cv2.Rodrigues(rotation)[0]
        external_matrix = np.hstack((R_rodrigues, translation))
        return external_matrix

    def get_screen_points(self, external_matrix):
        points = screen_points(self.camera_params, external_matrix, self.model_vertices)
        return points

    def pnp_rmse(self, lines):
        intersections = screen_lines_to_points(lines)
        external_matrix = self.get_external_matrix(intersections)
        points = self.get_screen_points(external_matrix)
        # if \
        # external_matrices_difference(external_matrix, self.tracker_params.gt2, self.camera_params, self.model_vertices)[
        #     2] < 0.03:
        #     cur_points = self.get_screen_points(external_matrix)
        #     frame = self.cur_frame.copy()
        #     for i, point in enumerate(cur_points):
        #         from screen_tracking.test.draw_result import to_screen
        #         cv2.circle(frame, to_screen(point), 3, (0, 0, 255), -1)
        #         cv2.line(frame, to_screen(cur_points[i]), to_screen(cur_points[(i + 1) % len(cur_points)]), (0, 0, 255))
        #     cv2.imshow('best', frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return np.linalg.norm(points - intersections)

    def previous_matrix_diff(self, lines, last_points):
        intersections = screen_lines_to_points(lines)
        previous_external_matrix = self.get_external_matrix(last_points)
        current_external_matrix = self.get_external_matrix(intersections)
        diff = external_matrices_difference(current_external_matrix, previous_external_matrix, self.camera_params,
                                            self.model_vertices)

    def best_rectangle(self, candidates):
        candidate_rectangles = list(itertools.product(*candidates))
        rmses = []
        rmses = np.array(list(map(self.pnp_rmse, candidate_rectangles)))
        indices = list(range(len(candidate_rectangles)))
        indices = sorted(indices, key=lambda x: rmses[x])
        return candidate_rectangles[indices[9]]

    def get_points(self, cur_frame, last_frame, last_points):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        last_lines = screen_points_to_lines(last_points)

        hough_frontier = HoughFrontier(self)
        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers = [RoFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers[0].show()
        result = self.best_rectangle(np.array(result))
        #
        intersections = screen_lines_to_points(result)
        # intersections = []
        # result = result[1:1]
        # self.show(result, cur_frame, intersections)

        return intersections

    def write_camera(self, tracking_result, pixels, frame):
        _, rotation, translation = cv2.solvePnP(self.model_vertices, pixels, self.camera_params, np.array([]),
                                                flags=self.tracker_params.PNP_FLAG)
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
            if frame_number == 2:
                break
            frame_number += 1
            # break

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file, tracker_params=None):
    tracker = Tracker(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
