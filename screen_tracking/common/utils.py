import os
from functools import lru_cache

import numpy as np
import pywavefront
import yaml

from screen_tracking.common.common import screen_points

# np.set_printoptions(precision=3, suppress=True)


class TrackingDataReader:
    DEFAULT_TEST = 'resources/tests/generated_tv_on0'
    DEFAULT_DESCRIPTION_FILE = 'test_description.yml'
    DEFAULT_VIDEO_OUTPUT = 'out.mov'
    DEFAULT_TRACKING_OUTPUT = 'tracking_result.yml'

    def __init__(self, **kwargs):
        self.test_directory = kwargs.get('test_directory', self.DEFAULT_TEST)
        self.test_description = kwargs.get('test_description', self.DEFAULT_DESCRIPTION_FILE)
        self.video_output = kwargs.get('video_output', self.DEFAULT_VIDEO_OUTPUT)
        self.tracking_result_output = kwargs.get('tracking_result_output', self.DEFAULT_TRACKING_OUTPUT)

    def relative_file(self, file):
        return os.path.join(self.test_directory, file)

    @lru_cache(None)
    def get_test_description(self):
        with open(self.relative_file(self.test_description)) as fin:
            result = yaml.load(fin, Loader=yaml.FullLoader)
        return result

    @lru_cache(None)
    def get_tracking_matrix(self, file):
        with open(self.relative_file(file)) as fin:
            input_frames = yaml.load(fin, Loader=yaml.FullLoader)
            result = {}
            if 'broken' in input_frames[0]:
                return {'broken': True}
            for frame in input_frames:
                R = np.array(frame['pose']['R'])
                t = np.array(frame['pose']['t']).reshape((3, 1))
                result[frame['frame']] = np.hstack((R, t))
                if 'time' in frame:
                    result[frame['frame']] = {
                        'pose': result[frame['frame']],
                        'time': frame['time']
                    }
        return result

    def get_ground_truth(self):
        return self.get_tracking_matrix(self.get_test_description()['ground_truth'])

    def get_tracking_result_output(self):
        return self.get_tracking_matrix(self.tracking_result_output)

    @lru_cache(None)
    def get_model_vertices(self):
        return np.array(
            pywavefront.Wavefront(self.relative_file(self.get_test_description()['mesh'])).vertices)

    def get_projection_matrix(self):
        return np.array(self.get_test_description()['projection'])

    def get_sequence_file(self):
        return self.get_test_description()['sequence']

    def write_result(self, result):
        with open(self.relative_file(self.tracking_result_output), 'w') as fout:
            to_write = []
            if 'broken' in result:
                to_write.append({'broken': True})
            else:
                for frame, frame_data in result.items():
                    matrix = frame_data['pose']
                    frame_result = {
                        'frame': frame,
                        'pose': {
                            'R': matrix[:3, :3].tolist(),
                            't': matrix[:3, 3].T.reshape(3).tolist()
                        },
                        'time': frame_data['time']
                    }
                    to_write.append(frame_result)

            print(to_write)
            yaml.dump(to_write, fout, default_flow_style=None)
            fout.close()

    def get_screen_points(self, frames, poses):
        result_screen_points = {}
        for frame in frames:
            if frame not in poses:
                continue
            result_screen_points[frame] = screen_points(
                self.get_projection_matrix(),
                poses[frame],
                self.get_model_vertices()
            )
        return result_screen_points

    def generate_user_input_from_ground_truth(self):
        resulting_points = self.get_screen_points(list(range(102)), self.get_ground_truth())
        return resulting_points

    def user_input(self):
        ground_truth_file = self.get_test_description()['ground_truth']
        result = None
        with open(self.relative_file(ground_truth_file)) as fin:
            content = yaml.load(fin, Loader=yaml.FullLoader)
            if 'pixels' in content:
                result = {1: np.array(content['pixels'], dtype=np.float)}
        if result is None:
            result = self.generate_user_input_from_ground_truth()
        return result

    def compare_input(self):
        return \
            self.get_model_vertices(), \
            self.get_projection_matrix(), \
            self.get_ground_truth(), \
            self.get_tracking_result_output(),

    def tracker_input(self):
        return \
            self.get_model_vertices(), \
            self.get_projection_matrix(), \
            self.relative_file(self.get_sequence_file()), \
            self.user_input(), \
            self.write_result, \
            self.relative_file(self.tracking_result_output)

    def draw_input(self):
        return \
            self.get_model_vertices(), \
            self.get_projection_matrix(), \
            self.relative_file(self.get_sequence_file()), \
            self.get_tracking_result_output(), \
            self.relative_file(self.video_output)

    def show_input(self):
        return self.relative_file(self.video_output)
