import cv2
import os
import yaml
import copy
import shutil

tests_dir = '../../resources/tests/'
suffix = 'tv_on'
initial_test_name = 'generated_' + suffix
initial_test_dir = tests_dir + initial_test_name + '/'

video_source = initial_test_dir + 'input.mp4'

cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

directories = [initial_test_dir[:-1] + str(i) + '/' for i in range(10)]
for dir in directories:
    try:
        os.mkdir(dir)
    except:
        pass
resulting_sources = [dir + 'input.mp4' for dir in directories]


outs = [cv2.VideoWriter(res_source, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for res_source in resulting_sources]

frame_number = 0
current_video = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_number += 1
        outs[current_video].write(frame)
        if frame_number % 10 == 0 and frame_number < 100: # TODO dirty hack
            current_video += 1
    else:
        break


td_filename = 'test_description.yml'
td_file = initial_test_dir + td_filename
td_outs = [dir + td_filename for dir in directories]



with open(td_file) as fin:
    test_description = yaml.load(fin, Loader=yaml.FullLoader)

tdscs = []

for i in range(10):
    tdsc = copy.deepcopy(test_description)
    if '..' in test_description['ground_truth']:
        tdsc['ground_truth'] = '../generated_tv_on' + str(i) + '/ground_truth.yml'
        tdsc['mesh'] = '../generated_tv_on' + str(i) + '/tv_picture_centered.obj'
    tdscs.append(tdsc)

for tdsc, td_out in zip(tdscs, td_outs):
    yaml.dump(tdsc, open(td_out, 'w'), default_flow_style=None)

for out in outs:
    out.release()
cap.release()


if 'on' in suffix:
    for dir in directories:
        shutil.copy(initial_test_dir + 'tv_picture_centered.obj', dir)



gt_filename = 'ground_truth.yml'
gt_file = initial_test_dir + gt_filename
gt_outs = [dir + gt_filename for dir in directories]

if 'on' in suffix:
    with open(gt_file) as fin:
        ground_truth = yaml.load(fin, Loader=yaml.FullLoader)
    for i, output in enumerate(gt_outs):
        if i + 1 != 10:
            gt_current = ground_truth[i * 10: (i + 1) * 10]
        else:
            gt_current = ground_truth[i * 10:]
        yaml.dump(gt_current, open(output, 'w'), default_flow_style=None)
