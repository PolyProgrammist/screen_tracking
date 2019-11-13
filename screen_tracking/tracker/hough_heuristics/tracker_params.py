import cv2


class TrackerParams:
    MARGIN_FRACTION = 1 / 5
    CANNY_THRESHOLD_1 = 20
    CANNY_THRESHOLD_2 = 40
    MIN_LINE_LENGTH = 50
    MAX_LINE_GAP = 1000
    APERTURE_SIZE = 3
    PNP_FLAG = cv2.SOLVEPNP_ITERATIVE
    L2_GRADIENT = True
    THRESHOLD_HOUGH_LINES_P = 30
    HOUGH_DISTANCE_RESOLUTION = 1
    HOUGH_ANGLE_RESOLUTION = 1 / 3
    MAX_GRADIENT = 30
    MAX_DIFF_GRADIENT_RECT = 5

    MAX_DIFF_PHI = 0.1
    MAX_DIFF_RO_INNER = 30
    MAX_DIFF_RO_OUTER = 60
    MAX_DIFF_PNP = 6
    MAX_DIFF_SQUARE = 0.03
    MAX_DIFF_PREVIOUS_POSE = 5
    MAX_DIFF_IN_OUT_PHI = 0.06
    MAX_DIFF_IN_OUT_DISTANCE = 40
    SMALLEST_SCREEN_FRAME = 3.0
    LARGEST_SCREEN_FRAME = 40


class TrackerState:
    pass
