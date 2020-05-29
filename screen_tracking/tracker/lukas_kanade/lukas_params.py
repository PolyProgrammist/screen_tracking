import cv2


class TrackerParams:
    MARGIN_FRACTION = 1 / 5
    PNP_FLAG = cv2.SOLVEPNP_ITERATIVE
