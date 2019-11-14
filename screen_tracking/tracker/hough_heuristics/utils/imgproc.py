import cv2
import numpy as np


def processed_sobel(frame, x, y):
    ksize = 3
    frame = cv2.Sobel(frame, cv2.CV_64F, x, y, ksize=ksize)
    frame = np.sum(frame, axis=2)

    # TODO: try morphology
    # sobel = cv2.morphologyEx(sobel, cv2.MORPH_OPEN, kernel=np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ]))

    # TODO: move magic number to tracker parameters
    ret, sobel_positive = cv2.threshold(frame, np.max(frame) * 0.05, np.max(frame), cv2.THRESH_TOZERO)
    frame = -frame
    ret, sobel_negative = cv2.threshold(frame, np.max(frame) * 0.05, np.max(frame), cv2.THRESH_TOZERO)
    sobel_negative = -sobel_negative
    frame = sobel_negative + sobel_positive

    return frame
