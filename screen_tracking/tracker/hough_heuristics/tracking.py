import logging

import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.frontiers.in_out_frontiers.distance_in_out_frontier import \
    DistanceInOutFrontier
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.frontiers import (
    show_best,
    PreviousPoseFrontier,
    PNPrmseFrontier,
    RoFrontier,
    PhiFrontier,
    HoughFrontier,
    RectFrontier,
    InOutFrontier, PhiInOutFrontier, SquareFrontier, GroundTruthFrontier, RectFromInOutFrontier, RectUniqueFrontier)
from screen_tracking.tracker.hough_heuristics.utils import (
    get_screen_points,
    screen_lines_to_points,
    screen_points_to_lines,
    draw_line)
from screen_tracking.tracker.hough_heuristics.utils.geom2d import polyarea


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
            show_best(side_frontiers[i], frame=frame, no_show=i < 3, max_count=7)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        self.state.predict_matrix = predict_matrix
        last_lines = screen_points_to_lines(last_points)

        hough_frontier = HoughFrontier(self)

        # show_best(hough_frontier)

        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers_out = [RoFrontier(frontier, last_line, inner=False) for frontier, last_line in
                              zip(side_frontiers, last_lines)]
        side_frontiers_in = [RoFrontier(frontier, last_line, inner=True) for frontier, last_line in
                             zip(side_frontiers, last_lines)]

        # self.show_list_best(side_frontiers_in)

        in_frontier = RectFrontier(side_frontiers_in)
        in_frontier = PreviousPoseFrontier(in_frontier)
        in_frontier = PNPrmseFrontier(in_frontier)
        in_frontier = SquareFrontier(in_frontier)
        # in_frontier = GroundTruthFrontier(in_frontier)
        # show_best(in_frontier, show_all=True)
        # in_frontier.candidates = in_frontier.top_current()[:1]

        out_frontier = RectFrontier(side_frontiers_out)
        out_frontier = PreviousPoseFrontier(out_frontier)
        out_frontier = PNPrmseFrontier(out_frontier)

        # show_best(out_frontier, show_all=True)

        print('in_frontier: ', len(in_frontier.top_current()))
        print('out_frontier: ', len(out_frontier.top_current()))

        in_out_frontier = InOutFrontier(in_frontier, out_frontier)
        print('in_out_frontier: ', len(in_out_frontier.top_current()))
        in_out_frontier = DistanceInOutFrontier(in_out_frontier)
        print('in_out_frontier: ', len(in_out_frontier.top_current()))
        # in_out_frontier = PhiInOutFrontier(in_out_frontier)
        print('in_out_frontier: ', len(in_out_frontier.top_current()))

        rect_frontier = RectFromInOutFrontier(in_out_frontier)
        rect_frontier = RectUniqueFrontier(rect_frontier)
        print('len of unique: ', len(rect_frontier.top_current()))
        rect_frontier = PNPrmseFrontier(rect_frontier)
        # rect_frontier = GroundTruthFrontier(rect_frontier)

        # show_best(rect_frontier)
        #
        # rect = rect_frontier.top_current()[0]
        # for i, t in enumerate(in_out_frontier.top_current()):
        #     if rect.lines == t.inner.lines:
        #         show_best(in_out_frontier, starting_point=i)
        #         break

        rect_frontier = rect_frontier.top_current()[0]
        lines = [candidate.line for candidate in rect_frontier.lines]
        intersections = screen_lines_to_points(lines)
        return intersections
        # return 0

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
        initial_frame_number = 90
        cap = cv2.VideoCapture(self.video_source)
        frame_number = initial_frame_number
        last_frame = 0
        for i in range(frame_number):
            ret, last_frame = cap.read()
        last_frame = [last_frame]
        last_points = [self.frame_pixels[frame_number]]
        self.write_camera(tracking_result, last_points[0], frame_number)
        predict_matrix = tracking_result[frame_number]

        while cap.isOpened():
            try:
                # if frame_number == initial_frame_number + 1:
                #     break
                frame_number += 1
                print('\nFrame number: ', frame_number)
                self.state.frame_number = frame_number
                # TODO: remove ground truth
                ret, frame = cap.read()
                if not ret:
                    break
                points = self.get_points(frame, last_frame[0], last_points[0], predict_matrix)
                self.write_camera(tracking_result, points, frame_number)

                _, rotation, translation = cv2.solvePnP(
                    self.model_vertices,
                    self.frame_pixels[frame_number],
                    self.camera_params,
                    np.array([]),
                    flags=self.tracker_params.PNP_FLAG
                )
                R_rodrigues = cv2.Rodrigues(rotation)[0]
                external_matrix = np.hstack((R_rodrigues, translation))

                predict_matrix = external_matrix  # tracking_result[frame_number]
                predicted_points = get_screen_points(self, predict_matrix)
                last_points[0] = predicted_points
                last_frame[0] = frame
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
