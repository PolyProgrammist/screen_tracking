import cv2

from screen_tracking.common.utils import TrackingDataReader
from screen_tracking.common.common import screen_points


def to_screen(point):
    return tuple(map(int, point))


def write_result(model_vertices, projection_matrix, video_source, tracking_result, output_file):
    frame_number = 0
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_number += 1

            if frame_number in tracking_result:
                points = screen_points(projection_matrix, tracking_result[frame_number], model_vertices)
                for i, point in enumerate(points):
                    cv2.circle(frame, to_screen(point), 3, (0, 0, 255), -1)
                    cv2.line(frame, to_screen(points[i]), to_screen(points[(i + 1) % len(points)]), (0, 0, 255))

            out.write(frame)
        else:
            break

    out.release()
    cap.release()


if __name__ == "__main__":
    reader = TrackingDataReader()
    write_result(*reader.draw_input())
