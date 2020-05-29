import cv2
import yaml

from screen_tracking.tracker.hough_heuristics.utils import screen_points_to_lines
from screen_tracking.tracker.hough_heuristics.utils.geom2d import aspect_ratio

known_aspect = True
test = '../../../resources/tests/despadown/'

pixels_file = test + 'ground_truth.yml'
model_file = test + 'model.obj'
video_source = test + 'sequence.mp4'
aspect_ratio_file = test + 'aspect_ratio.txt'

points = []

print(video_source)


def click_and_crop(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        points.append([x, y])


cap = cv2.VideoCapture(video_source)
ret, image = cap.read()
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        break

cv2.destroyAllWindows()

assert len(points) == 4

with open(pixels_file, 'w') as fout:
    yaml.dump({'pixels': points}, fout)

lines = screen_points_to_lines(points)
aspect = aspect_ratio(lines)
if known_aspect:
    aspect = 1.7777

height = 1
width = aspect * height

obj_file = """## OBJ file generated by Nuke ##

# vertex list - offset=0
v -{0} -{1} 0.000000
v {0} -{1} 0.000000
v {0} {1} 0.000000
v -{0} {1} 0.000000

# vertex texture coordinates - offset=0
vt 0.000000 -0.000000
vt 0 0
vt 0 0
vt 0.000000 1.000000

# vertex normals - offset=0
vn 0.000000 0.000000 1.000000
vn 0.000000 0.000000 1.000000
vn 0.000000 0.000000 1.000000
vn 0.000000 0.000000 1.000000

f 1/1/1 2/2/2 3/3/3 4/4/4

# end of file""".format(width, height)

with open(model_file, 'w') as fout:
    fout.write(obj_file)

with open(aspect_ratio_file, 'w') as fout:
    fout.write(str(aspect))

print('Aspect ratio: ', aspect)
