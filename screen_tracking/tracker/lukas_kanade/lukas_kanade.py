import logging

import numpy as np
import cv2

from screen_tracking.tracker.common_tracker import Tracker
from screen_tracking.tracker.hough_heuristics.utils import get_screen_points, screen_points_to_lines, draw_line
from screen_tracking.tracker.hough_heuristics.utils.draw import show_frame
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
