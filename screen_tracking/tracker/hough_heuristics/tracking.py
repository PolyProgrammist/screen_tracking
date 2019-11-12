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
    InOutFrontier, PhiInOutFrontier)
from screen_tracking.tracker.hough_heuristics.utils import (
    get_screen_points,
    screen_lines_to_points,
    screen_points_to_lines,
    draw_line)


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
            show_best(side_frontiers[i], frame=frame, no_show=i < 3, max_count=1)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        # last_points[0][1] += 1
        # lines = screen_points_to_lines(last_points)
        # for line in lines:
        #     draw_line(cur_frame, line)
        # cv2.imshow('screen', cur_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        self.state.predict_matrix = predict_matrix
        last_lines = screen_points_to_lines(last_points)

        hough_frontier = HoughFrontier(self)

        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        side_frontiers = [RoFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]

        rect_frontier = RectFrontier(side_frontiers)
        rect_frontier = PreviousPoseFrontier(rect_frontier)
        rect_frontier = PNPrmseFrontier(rect_frontier)

        print(len(rect_frontier.top_current()))
        in_out_frontier = InOutFrontier(rect_frontier)
        print(len(in_out_frontier.top_current()))
        in_out_frontier = PhiInOutFrontier(in_out_frontier)
        print(len(in_out_frontier.top_current()))
        in_out_frontier = DistanceInOutFrontier(in_out_frontier)
        print(len(in_out_frontier.top_current()))
        ll = len(in_out_frontier.top_current())
        # great = 0
        # number = 0
        # for candidate in in_out_frontier.top_current():
        #     show_best(in_out_frontier, starting_point=number)
        #     great += 1
        #     number += 1
        # print('great: ', great)
        # exit(0)

        # resulting_rect = rect_frontier.top_current()[0]
        in_out_rects = in_out_frontier.top_current()[0]
        lines = [candidate.line for candidate in in_out_rects.inner.lines]
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
                points = self.get_points(frame, last_frame[0], last_points[0], predict_matrix)
                self.write_camera(tracking_result, points, frame_number)

                predict_matrix = tracking_result[frame_number]
                predicted_points = get_screen_points(self, predict_matrix)
                last_points[0] = predicted_points
                last_frame[0] = frame
                frame_number += 1
                print(frame_number)
                # if frame_number == :
                #     break
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
