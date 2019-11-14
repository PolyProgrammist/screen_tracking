import cv2

from screen_tracking.tracker.hough_heuristics.utils.geom2d import screen_lines_to_points, screen_points_to_lines, \
    intersected_lines


def cut(frame, bbox):
    min_x, min_y, max_x, max_y = bbox
    return frame.copy()[min_y:max_y, min_x:max_x, :]


def draw_line(frame, line, color=(0, 0, 255)):
    cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), color=color, thickness=1)


def draw_point(frame, point):
    cv2.circle(frame, tuple(point.astype(int)), 5, color=(0, 255, 0), thickness=-1)


def rectangle_draw(frame, candidate):
    lines = [candidate.line for candidate in candidate.lines]
    lines = intersected_lines(lines)
    for line in lines:
        draw_line(frame, line)
