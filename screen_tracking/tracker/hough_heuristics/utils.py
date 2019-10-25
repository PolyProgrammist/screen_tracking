import numpy as np


def get_bounding_box(cur_frame, last_points, margin_fraction):
    length = 0
    for i in range(4):
        for j in range(4):
            cur_length = np.linalg.norm(last_points[i] - last_points[j])
            length = max(length, cur_length)
    offset_len = length * margin_fraction
    dxs = [-1, 0, 1, 0]
    dys = [0, 1, 0, -1]
    offsets = [[offset_len * dx, offset_len * dy] for dx, dy in zip(dxs, dys)]
    points = [point + offset for offset in offsets for point in last_points]

    min_x = np.clip(min(p[0] for p in points), 0, cur_frame.shape[1])
    max_x = np.clip(max(p[0] for p in points), 0, cur_frame.shape[1])
    min_y = np.clip(min(p[1] for p in points), 0, cur_frame.shape[0])
    max_y = np.clip(max(p[1] for p in points), 0, cur_frame.shape[0])

    result = min_x, min_y, max_x, max_y

    result = tuple(map(int, result))

    return result


def cut(frame, bbox):
    min_x, min_y, max_x, max_y = bbox
    return frame.copy()[min_y:max_y, min_x:max_x, :]


def adjust_vector_abc(vector):
    return vector / np.linalg.norm(vector[:2])


def points_to_abc(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    ar = np.array([a, b, c])
    return ar


def adjusted_abc(line):
    return adjust_vector_abc(points_to_abc(line[0], line[1]))


def screen_lines_to_points(lines):
    intersections = []
    for i in range(-1, 3):
        intersections.append(lines_intersection(lines[(i + 4) % 4], lines[(i + 1) % 4]))
    return np.array(intersections)


def screen_points_to_lines(last_points):
    lines = []
    for i in range(4):
        lines.append([last_points[i], last_points[(i + 1) % 4]])
    return np.array(lines)


def lines_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])
