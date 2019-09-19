import math
import cv2
import numpy as np
import pywavefront

model_vertices_file = 'sources/tv_picture_centered.obj'
camera_params_file = 'sources/camera_params.txt'
first_frame_pixels_file = 'sources/first_frame_2d_pixels.txt'
video_source = 'sources/tv_on.mp4'

np.set_printoptions(precision=3, suppress=True)


def read_camera_params():
    lines = open(camera_params_file).readlines()
    for i, line in enumerate(lines):
        if 'opencv_proj_mat =' in line:
            index = i + 1
    lines = lines[index: index + 3]
    lines = [list(map(float, line.split())) for line in lines]
    return np.array(lines)


def read_first_frame_pixels():
    return np.array([
        tuple(map(float, line.strip().split())) for line in open(first_frame_pixels_file).readlines()
    ])


def to_screen(point):
    return tuple(map(int, point))


# Read input data
model_vertices = np.array(pywavefront.Wavefront(model_vertices_file).vertices)
camera_params = read_camera_params()
first_frame_pixels = read_first_frame_pixels()

# Find rotation and translation
_, rotation, translation = cv2.solvePnP(model_vertices, first_frame_pixels, camera_params, np.array([]))

# Find external parameters matrix
R_x = np.array([[1, 0, 0],
                [0, math.cos(rotation[0]), -math.sin(rotation[0])],
                [0, math.sin(rotation[0]), math.cos(rotation[0])]
                ])
R_y = np.array([[math.cos(rotation[1]), 0, math.sin(rotation[1])],
                [0, 1, 0],
                [-math.sin(rotation[1]), 0, math.cos(rotation[1])]
                ])
R_z = np.array([[math.cos(rotation[2]), -math.sin(rotation[2]), 0],
                [math.sin(rotation[2]), math.cos(rotation[2]), 0],
                [0, 0, 1]
                ])
R = np.dot(R_z, np.dot(R_y, R_x))
external_matrix = np.hstack((R, translation))
print("External matrix:\n", external_matrix)

# Find points on screen
projection_matrix = np.dot(camera_params, external_matrix)
model_vertices_matrix = np.hstack((model_vertices, np.ones((len(model_vertices), 1)))).T
points_homogeneous = np.dot(projection_matrix, model_vertices_matrix).T
points = points_homogeneous[:, :2]
points_w = points_homogeneous[:, 2:]
points = points / points_w

# Reading first video frame
cap = cv2.VideoCapture(video_source)
ret, img = cap.read()

for i, point in enumerate(points):
    cv2.circle(img, to_screen(point), 3, (0, 0, 255), -1)
    cv2.line(img, to_screen(points[i]), to_screen(points[(i + 1) % len(points)]), (0, 0, 255))

cv2.imshow('First frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
