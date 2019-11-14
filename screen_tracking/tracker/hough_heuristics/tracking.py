import logging

import cv2
import numpy as np

from screen_tracking.tracker.hough_heuristics.frontiers.line_frontiers.ground_truth_line_frontier import \
    GroundTruthLineFrontier
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.frontiers import (
    show_best,
    PreviousPoseFrontier,
    PNPrmseFrontier,
    PreviousLineDistanceFrontier,
    PhiFrontier,
    HoughFrontier,
    RectFrontier,
    InOutFrontier, PhiInOutFrontier, SquareFrontier, GroundTruthFrontier, RectFromInOutFrontier, RectUniqueFrontier,
    LineGradientFrontier, RectangleGradientFrontier, OuterVarianceFrontier, DistanceInOutFrontier, AspectRatioFrontier)
from screen_tracking.tracker.hough_heuristics.utils import (
    get_screen_points,
    screen_lines_to_points,
    screen_points_to_lines,
    draw_line)
from screen_tracking.tracker.hough_heuristics.utils.geom2d import polyarea


class Tracker:
    tracker_params = TrackerParams()

    FRAMES_NUMBER_TO_TRACK = np.inf
    INITIAL_FRAME = 1
    PREVIOUS_GROUND_TRUTH = False

    SHOW_EACH_SIDE = int(1e9)

    def line_frontiers(self):
        last_lines = screen_points_to_lines(self.state.last_points)

        hough_frontier = HoughFrontier(self)
        side_frontiers_out = [hough_frontier for _ in last_lines]

        side_frontiers_out = [PhiFrontier(frontier, last_line) for frontier, last_line in
                              zip(side_frontiers_out, last_lines)]
        side_frontiers_out = [PreviousLineDistanceFrontier(frontier, last_line, inner=False) for frontier, last_line in
                              zip(side_frontiers_out, last_lines)]

        side_frontiers_out = [LineGradientFrontier(frontier) for frontier in side_frontiers_out]

        side_frontiers_in = [PreviousLineDistanceFrontier(frontier, last_line, inner=True) for frontier, last_line in
                             zip(side_frontiers_out, last_lines)]

        # self.show_list_best(side_frontiers_out)

        return side_frontiers_in, side_frontiers_out

    def rectangle_frontiers(self, side_frontiers_in, side_frontiers_out):
        in_frontier = RectFrontier(side_frontiers_in)
        in_frontier.print_best()
        # in_frontier = GroundTruthFrontier(in_frontier)
        # show_best(in_frontier)

        in_frontier = PreviousPoseFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = PNPrmseFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = SquareFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = RectangleGradientFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = OuterVarianceFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = AspectRatioFrontier(in_frontier)
        in_frontier.print_best()

        out_frontier = RectFrontier(side_frontiers_out)
        out_frontier = PreviousPoseFrontier(out_frontier)
        out_frontier = PNPrmseFrontier(out_frontier)

        in_frontier.print_best(caption='In frontier')
        out_frontier.print_best(caption='Out frontier')
        return in_frontier, out_frontier

    def in_out_frontier(self, in_frontier, out_frontier):
        in_out_frontier = InOutFrontier(in_frontier, out_frontier)
        in_out_frontier.print_best()
        in_out_frontier = DistanceInOutFrontier(in_out_frontier)
        in_out_frontier.print_best()
        in_out_frontier = PhiInOutFrontier(in_out_frontier)
        in_out_frontier.print_best()
        return in_out_frontier

    def rectangle_after_in_out(self, in_out_frontier):
        rect_frontier = RectFromInOutFrontier(in_out_frontier)
        rect_frontier = RectUniqueFrontier(rect_frontier)
        rect_frontier.print_best()
        rect_frontier = PNPrmseFrontier(rect_frontier)
        return rect_frontier

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        self.state.predict_matrix = predict_matrix

        side_frontiers_in, side_frontiers_out = self.line_frontiers()
        if side_frontiers_in is None:
            return None
        in_frontier, out_frontier = self.rectangle_frontiers(side_frontiers_in, side_frontiers_out)
        if in_frontier is None:
            return None
        in_out_frontier = self.in_out_frontier(in_frontier, out_frontier)
        if out_frontier is None:
            return None
        rect_frontier = self.rectangle_after_in_out(in_out_frontier)
        if rect_frontier is None:
            return None
        # show_best(rect_frontier, show_all=True)

        rect_frontier = rect_frontier.top_current()[0]
        lines = [candidate.line for candidate in rect_frontier.lines]
        intersections = screen_lines_to_points(lines)
        return intersections

    def track(self):
        self.state.productivity = {}
        tracking_result = {}
        initial_frame_number = self.INITIAL_FRAME
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
                if frame_number == initial_frame_number + self.FRAMES_NUMBER_TO_TRACK:
                    break
                frame_number += 1
                print('\nFrame number: ', frame_number)
                self.state.frame_number = frame_number
                ret, frame = cap.read()
                if not ret:
                    break
                points = self.get_points(frame, last_frame[0], last_points[0], predict_matrix)
                if points is not None:
                    self.write_camera(tracking_result, points, frame_number)
                    predict_matrix = tracking_result[frame_number]

                # TODO: remove ground truth
                if self.PREVIOUS_GROUND_TRUTH:
                    _, rotation, translation = cv2.solvePnP(
                        self.model_vertices,
                        self.frame_pixels[frame_number],
                        self.camera_params,
                        np.array([]),
                        flags=self.tracker_params.PNP_FLAG
                    )
                    R_rodrigues = cv2.Rodrigues(rotation)[0]
                    external_matrix = np.hstack((R_rodrigues, translation))
                    predict_matrix = external_matrix

                predicted_points = get_screen_points(self, predict_matrix)
                last_points[0] = predicted_points
                last_frame[0] = frame
            except Exception as error:
                logging.error('Tracker broken')
                logging.exception(error)
                break

        print()
        for key, value in self.state.productivity.items():
            print(key, value[1])

        return tracking_result

    def show_list_best(self, side_frontiers):
        frame = self.state.cur_frame.copy()
        for i in range(4):
            show_best(side_frontiers[i], frame=frame, no_show=i < 3, max_count=self.SHOW_EACH_SIDE)

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

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels
        self.state = TrackerState()


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file, tracker_params=None):
    tracker = Tracker(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
