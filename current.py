import cv2

from screen_tracking.common.utils import TrackingDataReader
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

reader = TrackingDataReader('resources/tests/generated_tv_on', test_description='test_description.yml')
video_file = reader.relative_file(reader.get_sequence_file())
cap = cv2.VideoCapture(video_file)


def get_points(current_image, previous_image, previous_points):
    bbox = get_bounding_box(current_image, previous_points)
    current_image = cut(current_image, bbox)
    previous_image = cut(previous_image, bbox)
    previous_lines = lines(points)
    previous_hough_peaks = hough_parameters(previous_lines)
    current_hough_peaks_candidates = hough_line_peaks(current_image)
    current_hough_peaks_candidates = nearest(current_hough_peaks_candidates, previous_hough_peaks)
    current_hough_peaks_candidates = most_common_3(current_hough_peaks_candidates, previous_hough_peaks)
    current_hough_peaks = best_paralellogram(current_hough_peaks_candidates)
    current_points = points_from_lines()
    return current_points


def draw_lines(img, houghLines, thickness=2):
    color = [0, 0, 255]
    for line in houghLines:
        for rho, theta in line:
            cv2.circle(img, rho, theta, 3, color, thickness)


def go_frame(cnt):
    ret, frame = cap.read()
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    minLineLength = 100
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(edges, theta=tested_angles)
    # img = np.zeros()

    hough = np.log(1 + h)
    print(hough.shape)
    print(np.max(hough))
    hough /= np.max(hough)
    hough = hough * 255
    hough = hough.astype(np.uint8)

    hough_add = np.zeros(hough.shape, dtype=np.uint8)
    hough_add = cv2.cvtColor(hough_add, cv2.COLOR_GRAY2BGR)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.2 * np.max(h))):
        print(angle, _)
        y = _
        x = angle * 360 / np.math.pi
        cv2.circle(hough_add, (int(x), int(y)), 3, [0, 255, 0], 3)

    hough = cv2.cvtColor(hough, cv2.COLOR_GRAY2BGR)
    cv2.imshow('hough', hough)

    hough = cv2.resize(hough, (500, 500))

    cv2.imwrite('resources/tmp/hough' + str(cnt) + '.jpg', hough)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('resources/tmp/detected' + str(cnt) + '.jpg', img)


go_frame(1)
go_frame(2)

cv2.waitKey(0)
cv2.destroyAllWindows()
