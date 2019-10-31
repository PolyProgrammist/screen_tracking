import logging

import cv2
import numpy as np

from screen_tracking.common import TrackingDataReader
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.frontiers import (
    show_best,
    PreviousPoseFrontier,
    PNPrmseFrontier,
    RoFrontier,
    PhiFrontier,
    HoughFrontier,
    RectFrontier,
    RectangleShowKFrontier,
    GroundTruthFrontier
)
from screen_tracking.tracker.hough_heuristics.utils import (
    get_screen_points,
    screen_lines_to_points,
    screen_points_to_lines
)


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
            show_best(side_frontiers[i], frame=frame, no_show=i < 3, max_count=1, starting_point=1)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix, frame_number):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        self.state.predict_matrix = predict_matrix
        self.state.frame_number = frame_number
        last_lines = screen_points_to_lines(last_points)

        hough_frontier = HoughFrontier(self)

        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers = [RoFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]

        rect_frontier = RectFrontier(side_frontiers)
        rect_frontier = PreviousPoseFrontier(rect_frontier)
        rect_frontier = PNPrmseFrontier(rect_frontier)
        rect_frontier = GroundTruthFrontier(rect_frontier)
        print('First previous top = ', rect_frontier.candidates[0].previous_top)
        # rect_frontier = PNPrmseFrontier(rect_frontier)
        print([candidate.current_score_ for candidate in rect_frontier.top_current()])
        print(len(rect_frontier.top_current()))
        rect_frontier = RectangleShowKFrontier(rect_frontier)
        # print('Remain: ', len(rect_frontier.candidates))

        for i in range(0, 33):
            show_best(rect_frontier, colors=[(0, 0, 255)], starting_point=i, only_lines=True)

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
        tracking_result[frame] = external_matrix

    def track(self):
        tracking_result = {}

        cap = cv2.VideoCapture(self.video_source)

        ret, last_frame = cap.read()
        last_frame = [last_frame]
        last_points = [self.frame_pixels[1]]
        frame_number = 1
        self.write_camera(tracking_result, last_points[0], frame_number)
        predict_matrix = tracking_result[frame_number]
        frame_number = 2

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                points = self.get_points(frame, last_frame[0], last_points[0], predict_matrix, frame_number)
                self.write_camera(tracking_result, points, frame_number)

                reader = TrackingDataReader()
                predict_matrix = reader.get_ground_truth()[frame_number]
                # predict_matrix = tracking_result[frame_number]
                predicted_points = get_screen_points(self, predict_matrix)
                last_points[0] = predicted_points
                print('Frame number: ', frame_number)
                last_frame[0] = frame
                if frame_number == 100:
                    break
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
