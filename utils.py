import yaml
import numpy as np
import pywavefront
from functools import lru_cache
from common import screen_points

np.set_printoptions(precision=3, suppress=True)


class TrackingDataReader:
    def __init__(self, test='sources/'):
        self.__test = test

    @lru_cache(None)
    def get_test_description(self):
        with open(self.__test + 'test_description.yml') as fin:
            return yaml.load(fin)

    @lru_cache(None)
    def get_model_vertices(self):
        return np.array(pywavefront.Wavefront(self.__test + self.get_test_description()['mesh']).vertices)

    @lru_cache(None)
    def get_tracking_matrix(self, file):
        with open(self.__test + file) as fin:
            input = yaml.load(fin)
            result = {}
            for frame in input:
                R = np.array(frame['pose']['R'])
                t = np.array(frame['pose']['t']).reshape((3,1))
                result[frame['frame']] = np.hstack((R, t))
            return result

    def get_ground_truth(self):
        return self.get_tracking_matrix(self.get_test_description()['ground_truth'])

    def get_tracking_result(self):
        return self.get_tracking_matrix('tracking_result.yml')

    def get_projection_matrix(self):
        return np.array(self.get_test_description()['projection'])

    def tracker_input(self):
        return self.get_model_vertices(), self.get_projection_matrix(), self.generate_user_input()

    def get_resulting_points(self, frames, poses):
        result = {}
        for frame in frames:
            result[frame] = screen_points(
                self.get_projection_matrix(),
                poses[frame],
                self.get_model_vertices()
            )
        return result

    def generate_user_input(self):
        return self.get_resulting_points([1, 5, 15, 23], self.get_ground_truth())


reader = TrackingDataReader()
