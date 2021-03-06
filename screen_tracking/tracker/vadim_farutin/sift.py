import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hog

from screen_tracking.tracker.vadim_farutin.utils_vadim_farutin import VadimFarutinTrackerAdapter


class SiftTrackerVadimFarutin:
    OCTAVES_NUMBER = 4
    SAMPLES_PER_OCTAVE = 5
    SCALE_SPACE_SIGMA = 1 / math.sqrt(2)
    SCALE_SPACE_FACTOR = math.sqrt(2)
    CONTRAST_THRESHOLD = 100
    CORNER_THRESHOLD = 10

    def __init__(self, camera_mat, screen, frame_size):
        self.frame_size = frame_size
        self.camera_mat = camera_mat
        self.object_points = screen

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        print('track frame started')

        octaves = SiftTrackerVadimFarutin.generate_differences_of_gaussians(
            frame2_grayscale_mat)

        extrema = []

        for octave in octaves:
            height, width = octave[0].shape
            scale_x = self.frame_size[0] / width
            scale_y = self.frame_size[1] / height

            for i in range(1, len(octave) - 1):
                Hxx, Hxy, Hyy = hessian_matrix(octave[i], order='rc')
                histograms = SiftTrackerVadimFarutin.histograms(octave[i])

                for x in range(1, width - 1):
                    for y in range(1, height - 1):
                        hessian = [Hxx[y][x], Hxy[y][x], Hyy[y][x]]

                        is_keypoint = SiftTrackerVadimFarutin.is_keypoint(octave[i - 1],
                                                              octave[i],
                                                              octave[i + 1],
                                                              x, y,
                                                              hessian)

                        if is_keypoint:
                            histogram = histograms[y][x][0][0]
                            orientation = np.argmax(histogram)
                            descriptor = SiftTrackerVadimFarutin.generate_descriptor(
                                octave[i], x, y)

                            extrema.append([x * scale_x, y * scale_y,
                                            scale_x, scale_y,
                                            orientation,
                                            descriptor])

        # Keypoint localization, find subpixel extrema
        # Keypoint descriptor
        # Match features

        # keypoint_image = cv2.cvtColor(frame2_grayscale_mat, cv2.COLOR_GRAY2BGR)
        # for keypoint in extrema:
        #     cv2.circle(keypoint_image,
        #                (int(keypoint[0]), int(keypoint[1])),
        #                1, (0, 0, 255), 1)
        # while True:
        #     cv2.imshow('keypoints', keypoint_image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         cv2.destroyAllWindows()
        #         break

        pos2_rotation_mat = pos1_rotation_mat
        pos2_translation = pos1_translation

        print('track frame finished')

        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def L(image, sigma):
        return gaussian_filter(image, sigma)

    @staticmethod
    def generate_differences_of_gaussians(image):
        octaves = []
        initial_image = np.copy(image)

        for i in range(SiftTrackerVadimFarutin.OCTAVES_NUMBER):
            samples = []
            differences = []
            sigma = SiftTrackerVadimFarutin.SCALE_SPACE_SIGMA
            current_image = []

            for j in range(SiftTrackerVadimFarutin.SAMPLES_PER_OCTAVE):
                previous_image = np.copy(current_image)
                current_image = SiftTrackerVadimFarutin.L(initial_image, sigma)
                samples.append(current_image)
                sigma *= SiftTrackerVadimFarutin.SCALE_SPACE_FACTOR

                if len(previous_image) != 0:
                    differences.append(current_image - previous_image)
                    # while True:
                    #     cv2.imshow(str(sigma), differences[-1])
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         cv2.destroyAllWindows()
                    #         break

            octaves.append(differences)
            width, height = initial_image.shape
            initial_image = cv2.resize(initial_image, (height // 2, width // 2))

        return octaves

    @staticmethod
    def histograms(image):
        return hog(image,
                   orientations=36,
                   pixels_per_cell=(1, 1),
                   cells_per_block=(1, 1),
                   block_norm='L1',
                   feature_vector=False)

    @staticmethod
    def neighbours(lower, middle, higher, x, y):
        return np.concatenate((lower[y - 1:y + 2, x - 1:x + 2],
                               middle[y - 1:y + 2, x - 1:x + 2],
                               higher[y - 1:y + 2, x - 1:x + 2]),
                              axis=None)

    @staticmethod
    def is_extrema(lower, middle, higher, x, y):
        neighbours = SiftTrackerVadimFarutin.neighbours(lower, middle, higher, x, y)
        minima = min(neighbours)
        maxima = max(neighbours)
        point_value = middle[y][x]

        unique, counts = np.unique(neighbours, return_counts=True)
        count_dict = dict(zip(unique, counts))

        return (point_value == minima or point_value == maxima) \
            and count_dict[point_value] == 1

    @staticmethod
    def contrast_threshold_passed(image, x, y):
        return abs(image[y][x]) > SiftTrackerVadimFarutin.CONTRAST_THRESHOLD

    @staticmethod
    def is_corner(hxx, hxy, hyy):
        threshold_value = (SiftTrackerVadimFarutin.CORNER_THRESHOLD + 1) ** 2 \
                          / SiftTrackerVadimFarutin.CORNER_THRESHOLD
        trace = hxx + hyy
        det = hxx * hyy - hxy * hxy

        return det < 0 or trace ** 2 < threshold_value * det

    @staticmethod
    def is_keypoint(lower, middle, higher, x, y, hessian):
        hxx, hxy, hyy = hessian[0], hessian[1], hessian[2]

        is_contrast = SiftTrackerVadimFarutin.contrast_threshold_passed(middle, x, y)
        if not is_contrast:
            return False

        is_corner = SiftTrackerVadimFarutin.is_corner(hxx, hxy, hyy)
        if not is_corner:
            return False

        is_extrema = SiftTrackerVadimFarutin.is_extrema(lower, middle, higher, x, y)
        return is_extrema

    @staticmethod
    def generate_descriptor(image, x, y):
        return []


class SiftTracker(VadimFarutinTrackerAdapter):
    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        super().__init__(model_vertices, camera_params, video_source, frame_pixels)

    def get_tracker_class(self):
        return SiftTrackerVadimFarutin