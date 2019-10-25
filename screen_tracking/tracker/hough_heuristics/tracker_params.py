import cv2


class TrackerParams:
    MARGIN_FRACTION = 1 / 5
    CANNY_THRESHOLD_1 = 50
    CANNY_THRESHOLD_2 = 150
    MIN_LINE_LENGTH = 50
    MAX_LINE_GAP = 30
    THRESHOLD_HOUGH_LINES_P = 50
    APERTURE_SIZE = 3
    PNP_FLAG = cv2.SOLVEPNP_ITERATIVE


class TrackerState:
    pass
