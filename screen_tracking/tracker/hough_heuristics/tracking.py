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
        # print('hough ', len(hough_frontier.top_current()))
        side_frontiers = [hough_frontier for _ in last_lines]
        side_frontiers = [PhiFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]
        # print('phi ', [len(side.top_current()) for side in side_frontiers])
        # phi_scores = np.array([candidate.current_score_ for candidate in side_frontiers[0].top_current()])

        side_frontiers = [RoFrontier(frontier, last_line) for frontier, last_line in zip(side_frontiers, last_lines)]

        if self.state.frame_number == 3:
            show_best(side_frontiers[3])
        # print('ro ', [len(side.top_current()) for side in side_frontiers])
        # ro_scores = np.array([candidate.current_score_ for candidate in side_frontiers[0].top_current()])

        # ln = min(len(ro_scores), len(phi_scores))
        # diff_scores = (ro_scores[:ln] / phi_scores[:ln]).mean()
        # print(ro_scores )
        # print(phi_scores)
        # print('average diff', diff_scores)

        rect_frontier = RectFrontier(side_frontiers)

        # print('before pnprmse', len(rect_frontier.top_current()))

        # rect_frontier = GroundTruthFrontier(rect_frontier)
        # rect_frontier.candidates = rect_frontier.top_current(max_count=1)
        rect_frontier = PNPrmseFrontier(rect_frontier)
        if self.state.frame_number == 3:
            show_best(rect_frontier)

        # b = {'lines': [
        #     {'phi': 0.012413998605904641, 'ro': 7.986038901179029},
        #     {'phi': 0.008950050904575235, 'ro': 0.9928337855714062},
        #     {'phi': 0.0029629066700249673, 'ro': 3.7704390801020793},
        #     {'phi': 0.0015595882971870534, 'ro': 0.14911804754268587}],
        #     'ground_truth': 0.034702612471380644, 'pnprmse': 1.8612209999792495, 'previous_pose': 0.04412731425847862}
        #
        # a = {'lines': [
        #     {'phi': 0.012413998605904641, 'ro': 7.986038901179029},
        #     {'phi': 0.003927907745674197, 'ro': 4.753696102474578},
        #     {'phi': 0.0029629066700249673, 'ro': 3.7704390801020793},
        #     {'phi': 0.010437384051952048, 'ro': 1.509483078328742}],
        #     'pnprmse': 1.9281745813911357, 'previous_pose': 0.03665580457104706}

        # print('before previous', len(rect_frontier.top_current()))
        rect_frontier = PreviousPoseFrontier(rect_frontier)

        # print([candidate.scores for candidate in rect_frontier.top_current()][0])

        # print(rect_frontier.top_current()[2].current_score_)

        print(len(rect_frontier.top_current()))
        resulting_rect = rect_frontier.top_current()[0]
        lines = [candidate.line for candidate in resulting_rect.lines]
        intersections = screen_lines_to_points(lines)

        return intersections

    def write_camera(self, tracking_result, pixels, frame):
        _, rotation, translation = cv2.solvePnP(self.model_vertices, pixels, self.camera_params, np.array([]),
                                                flags=self.tracker_params.PNP_FLAG)
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
            # try:
            self.state.frame_number = frame_number
            ret, frame = cap.read()
            if not ret:
                break
            points = self.get_points(frame, last_frame[0], last_points[0])
            self.write_camera(tracking_result, points, frame_number)
            predict_matrix = predict_next(tracking_result[frame_number - 1], tracking_result[frame_number - 1])
            # print(predict_matrix)
            predicted_points = get_screen_points(self, predict_matrix)
            last_points[0] = predicted_points
            last_frame[0] = frame
            print(frame_number)
            # if frame_number == 2:
            #     break
            frame_number += 1
        # except:
        #     logging.error('Tracker broken')
        #     break

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file, tracker_params=None):
    tracker = Tracker(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
