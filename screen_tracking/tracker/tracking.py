import cv2
import numpy as np

from screen_tracking.common.utils import TrackingDataReader


class Tracker:
    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        self.model_vertices = model_vertices
        self.camera_params = camera_params
        self.video_source = video_source
        self.frame_pixels = frame_pixels

    def track(self):
        tracking_result = {}

        # cap = cv2.VideoCapture(video_source)

        for frame, pixels in self.frame_pixels.items():
            _, rotation, translation = cv2.solvePnP(self.model_vertices, pixels, self.camera_params, np.array([]))
            R_rodrigues = cv2.Rodrigues(rotation)[0]
            external_matrix = np.hstack((R_rodrigues, translation))
            tracking_result[frame] = external_matrix

        return tracking_result


def track(model_vertices, camera_params, video_source, frame_pixels, write_callback, result_file):
    result = Tracker(model_vertices, camera_params, video_source, frame_pixels).track()
    write_callback(result)


if __name__ == "__main__":
    reader = TrackingDataReader()
    track(*reader.tracker_input())
