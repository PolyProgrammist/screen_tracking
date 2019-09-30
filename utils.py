import yaml
import numpy as np
import pywavefront
from functools import lru_cache
from common import screen_points

np.set_printoptions(precision=3, suppress=True)


class TrackingDataReader:
    def __init__(self, test='sources/'):
        self.__test = test

    def get_file_root_relative(self, file):
        return self.__test + file

    @lru_cache(None)
    def get_test_description(self):
        with open(self.get_file_root_relative('test_description.yml')) as fin:
            return yaml.load(fin)

    @lru_cache(None)
    def get_model_vertices(self):
        return np.array(pywavefront.Wavefront(self.get_file_root_relative(self.get_test_description()['mesh'])).vertices)

    @lru_cache(None)
    def get_tracking_matrix(self, file):
        with open(self.get_file_root_relative(file)) as fin:
            input = yaml.load(fin)
            result = {}
            for frame in input:
                R = np.array(frame['pose']['R'])
                t = np.array(frame['pose']['t']).reshape((3,1))
                result[frame['frame']] = np.hstack((R, t))
            return result

    def write_result(self, result):
        with open(self.get_file_root_relative(self.get_tracking_result_file()), 'w') as fout:
            to_write = []
            for frame, matrix in result.items():
                frame_result = {
                    'frame': frame,
                    'pose': {
                        'R': matrix[:3, :3].tolist(),
                        't': matrix[:3, 3].T.reshape(3).tolist()
                    }
                }
                to_write.append(frame_result)

            yaml.dump(to_write, fout, default_flow_style=None)

    def get_video_source(self):
        return self.get_test_description()['source_to_track']

    def get_ground_truth(self):
        return self.get_tracking_matrix(self.get_test_description()['ground_truth'])

    def get_tracking_result_file(self):
        return 'tracking_result.yml'

    def get_tracking_result(self):
        return self.get_tracking_matrix(self.get_tracking_result_file())

    def get_projection_matrix(self):
        return np.array(self.get_test_description()['projection'])

    def compare_input(self):
        return \
            self.get_model_vertices(),\
            self.get_projection_matrix(),\
            self.get_ground_truth(),\
            self.get_tracking_result()

    def draw_input(self):
        return \
            self.get_model_vertices(),\
            self.get_projection_matrix(),\
            self.get_video_source(),\
            self.get_tracking_result()

    def tracker_input(self):
        return \
            self.get_model_vertices(), \
            self.get_projection_matrix(), \
            self.get_file_root_relative(self.get_video_source()), \
            self.generate_user_input_from_ground_truth(), \
            self.get_file_root_relative(self.get_tracking_result_file())

    def get_resulting_points(self, frames, poses):
        result = {}
        for frame in frames:
            result[frame] = screen_points(
                self.get_projection_matrix(),
                poses[frame],
                self.get_model_vertices()
            )
        return result

    def generate_user_input_from_ground_truth(self):
        resulting_points = self.get_resulting_points([1, 5, 15, 23], self.get_ground_truth())
        for frame, points in resulting_points.items():
            points += np.random.rand(len(points), 2) * 3
        return resulting_points
