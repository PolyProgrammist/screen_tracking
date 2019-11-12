import cv2


class TrackerParams:
    MARGIN_FRACTION = 1 / 5
    CANNY_THRESHOLD_1 = 30
    CANNY_THRESHOLD_2 = 70
    MIN_LINE_LENGTH = 150
    MAX_LINE_GAP = 100
    THRESHOLD_HOUGH_LINES_P = 40
    APERTURE_SIZE = 3
    PNP_FLAG = cv2.SOLVEPNP_ITERATIVE
    L2_GRADIENT = True

    MAX_DIFF_PHI = 0.1
    MAX_DIFF_RO_INNER = 30
    MAX_DIFF_RO_OUTER = 60
    MAX_DIFF_PNP = 5
    MAX_DIFF_SQUARE = 0.03
    MAX_DIFF_PREVIOUS_POSE = 5
    MAX_DIFF_IN_OUT_PHI = 0.05
    MAX_DIFF_IN_OUT_DISTANCE = 40
    SMALLEST_SCREEN_FRAME = 0.5
    LARGEST_SCREEN_FRAME = 40


class TrackerState:
    pass
