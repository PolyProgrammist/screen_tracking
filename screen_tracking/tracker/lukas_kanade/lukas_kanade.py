import logging

import numpy as np
import cv2

from screen_tracking.tracker.common_tracker import Tracker
from screen_tracking.tracker.hough_heuristics.utils import get_screen_points
from screen_tracking.tracker.lukas_kanade.lukas_params import TrackerParams


class LukasKanadeTracker(Tracker):
    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        super().__init__(model_vertices, camera_params, video_source, frame_pixels)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):

        last_points = np.array([[point] for point in last_points], dtype=np.float32)
        lk_params = dict(winSize=(70, 70),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))

        color = np.random.randint(0, 255, (100, 3))

        old_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        frame = cur_frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, last_points, None, **lk_params)
        good_new = p1[st == 1]
        return good_new

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
