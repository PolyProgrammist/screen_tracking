import logging

import cv2
import numpy as np
import time

from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerState
from screen_tracking.tracker.hough_heuristics.utils import get_screen_points


class Tracker:
    tracker_params = None

    FRAMES_NUMBER_TO_TRACK = np.inf
    INITIAL_FRAME = 1
    PREVIOUS_GROUND_TRUTH = False

    SHOW_EACH_SIDE = int(1e9)

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels
        self.state = TrackerState()

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
        tracking_speed = {}
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

        broken = False


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
                start_time = time.time()
                points = self.get_points(frame, last_frame[0], last_points[0], predict_matrix)
                end_time = time.time()
                if points is not None:
                    self.write_camera(tracking_result, points, frame_number)
                    tracking_speed[frame_number] = end_time - start_time
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
                print(error)
                logging.exception(error)
                broken = True
                break

        print()
        for key, value in self.state.productivity.items():
            print(key, value[1])

        if broken:
            return {'broken': True}
        return self.unite_tracking_result_and_tracking_time(tracking_result, tracking_speed)

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        pass

    def unite_tracking_result_and_tracking_time(self, tracking_result, tracking_speed):
        result = {}
        for frame, array in tracking_result.items():
            result[frame] = {'pose': array, 'time': tracking_speed.get(frame)}
        return result


def common_track(
        model_vertices,
        camera_params,
        video_source,
        frame_pixels,
        write_callback,
        result_file,
        tracker_type,
        tracker_params=None
):
    tracker = tracker_type(model_vertices, camera_params, video_source, frame_pixels)

    if tracker_params is not None:
        tracker.tracker_params = tracker_params

    result = tracker.track()
    write_callback(result)
    return result
