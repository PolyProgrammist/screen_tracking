import numpy as np
import yaml

model_vertices_file = 'tv_picture_centered.obj'
camera_params_file = 'camera_params.txt'
first_frame_pixels_file = 'first_frame_2d_pixels.txt'
external_parameters_ground_truth = 'result_object_animation.txt'
video_source = 'tv_on.mp4'


def read_camera_params():
    lines = open(camera_params_file).readlines()
    for i, line in enumerate(lines):
        if 'opencv_proj_mat =' in line:
            index = i + 1
    lines = lines[index: index + 3]
    lines = [list(map(float, line.split())) for line in lines]
    return np.array(lines)


result = {
    'mesh': model_vertices_file,
    'projection': read_camera_params().tolist(),
    'source_to_track': video_source,
    'ground_truth': 'ground_truth.yml'
}

with open('test_description.yml', 'w') as fout:
    yaml.dump(result, fout, default_flow_style=None)


def read_ground_truth():
    lines = open(external_parameters_ground_truth).readlines()
    result = []
    for index, line in enumerate(lines):
        if line.strip() == '':
            continue
        ground_truth_external_matrix = line.split()
        ground_truth_external_matrix = np.array(list(map(float, ground_truth_external_matrix))).reshape((4, 3)).T
        ground_truth_external_matrix[:3, :3] = ground_truth_external_matrix[:3, :3].T
        matrix = ground_truth_external_matrix
        rotation = matrix[:3, :3]
        translation = matrix[:, 3:4]
        index += 1
        result.append({
            'frame': index,
            'pose': {
                'R': rotation.tolist(),
                't': translation.reshape(3).tolist()
            }
        })
    return result


ground_truth = read_ground_truth()

with open('ground_truth.yml', 'w') as fout:
    yaml.dump(ground_truth, fout, default_flow_style=None)