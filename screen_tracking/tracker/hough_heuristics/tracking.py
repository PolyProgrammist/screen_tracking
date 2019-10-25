import logging
import math
import itertools

import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.frontiers.frontier import show_best
from screen_tracking.tracker.hough_heuristics.frontiers.ground_truth_frontier import GroundTruthFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.hough_frontier import HoughFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.phi_frontier import PhiFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.pnp_rmse_frontier import PNPrmseFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.previous_pose_frontier import PreviousPoseFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.rect_frontier import RectFrontier
from screen_tracking.tracker.hough_heuristics.frontiers.ro_frontier import RoFrontier
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.utils.geom2d import \
    screen_lines_to_points, screen_points_to_lines
from screen_tracking.tracker.hough_heuristics.utils.geom3d import predict_next, get_screen_points


class Tracker:
    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels
        self.state = TrackerState()

    def show_list_best(self, side_frontiers):
        frame = self.state.cur_frame.copy()
        for i in range(4):
            show_best(side_frontiers[i], frame=frame, no_show=i < 3)

    def get_points(self, cur_frame, last_frame, last_points):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        last_lines = screen_points_to_lines(last_points)

        hough_frontier = HoughFrontier(self)

        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers = [RoFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]

        rect_frontier = RectFrontier(side_frontiers)
        rect_frontier = PNPrmseFrontier(rect_frontier)
        rect_frontier = PreviousPoseFrontier(rect_frontier)

        resulting_rect = rect_frontier.top_current()[0]
        lines = [candidate.line for candidate in resulting_rect.lines]
        intersections = screen_lines_to_points(lines)

        return intersections

    def write_camera(self, tracking_result, pixels, frame):
        _, rotation, translation = cv2.solvePnP(
            self.model_vertices,
            pixels,
            self.camera_params,
            np.array([]),
            flags=self.tracker_params.PNP_FLAG
        )
        R_rodrigues = cv2.Rodrigues(rotation)[0]
        external_matrix = np.hstack((R_rodrigues, translation))
        self.last_matrix = external_matrix
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

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                points = self.get_points(frame, last_frame[0], last_points[0])
                self.write_camera(tracking_result, points, frame_number)

                predict_matrix = predict_next(tracking_result[frame_number - 1], tracking_result[frame_number - 1])
                predicted_points = get_screen_points(self, predict_matrix)
                last_points[0] = predicted_points
                last_frame[0] = frame
                frame_number += 1
            except Exception as error:
                logging.error('Tracker broken')
                logging.exception(error)
                break

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file, tracker_params=None):
    tracker = Tracker(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
