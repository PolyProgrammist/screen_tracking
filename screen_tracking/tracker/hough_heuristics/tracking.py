import logging

import cv2
import numpy as np

from screen_tracking.tracker.common_tracker import Tracker
from screen_tracking.tracker.hough_heuristics.frontiers import (
    show_best,
    PreviousPoseFrontier,
    PNPrmseFrontier,
    PreviousLineDistanceFrontier,
    PhiFrontier,
    HoughFrontier,
    RectFrontier,
    InOutFrontier, PhiInOutFrontier, SquareFrontier, RectFromInOutFrontier, RectUniqueFrontier,
    LineGradientFrontier, RectangleGradientFrontier, OuterVarianceFrontier, DistanceInOutFrontier, AspectRatioFrontier)
from screen_tracking.tracker.hough_heuristics.tracker_params import TrackerParams, TrackerState
from screen_tracking.tracker.hough_heuristics.utils import (
    get_screen_points,
    screen_lines_to_points,
    screen_points_to_lines)


class HoughTracker(Tracker):

    def line_frontiers(self):
        last_lines = screen_points_to_lines(self.state.last_points)

        hough_frontier = HoughFrontier(self)
        side_frontiers_out = [hough_frontier for _ in last_lines]

        side_frontiers_out = [PhiFrontier(frontier, last_line) for frontier, last_line in
                              zip(side_frontiers_out, last_lines)]
        side_frontiers_out = [PreviousLineDistanceFrontier(frontier, last_line, inner=False) for frontier, last_line in
                              zip(side_frontiers_out, last_lines)]

        side_frontiers_out = [LineGradientFrontier(frontier) for frontier in side_frontiers_out]

        side_frontiers_in = [PreviousLineDistanceFrontier(frontier, last_line, inner=True) for frontier, last_line in
                             zip(side_frontiers_out, last_lines)]

        # self.show_list_best(side_frontiers_out)

        return side_frontiers_in, side_frontiers_out

    def rectangle_frontiers(self, side_frontiers_in, side_frontiers_out):
        in_frontier = RectFrontier(side_frontiers_in)
        in_frontier.print_best()
        in_frontier = PreviousPoseFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = PNPrmseFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = SquareFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = RectangleGradientFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = OuterVarianceFrontier(in_frontier)
        in_frontier.print_best()
        in_frontier = AspectRatioFrontier(in_frontier)
        in_frontier.print_best()

        out_frontier = RectFrontier(side_frontiers_out)
        out_frontier = PreviousPoseFrontier(out_frontier)
        out_frontier = PNPrmseFrontier(out_frontier)

        in_frontier.print_best(caption='In frontier')
        out_frontier.print_best(caption='Out frontier')
        return in_frontier, out_frontier

    def in_out_frontier(self, in_frontier, out_frontier):
        in_out_frontier = InOutFrontier(in_frontier, out_frontier)
        in_out_frontier.print_best()
        in_out_frontier = DistanceInOutFrontier(in_out_frontier)
        in_out_frontier.print_best()
        in_out_frontier = PhiInOutFrontier(in_out_frontier)
        in_out_frontier.print_best()
        return in_out_frontier

    def rectangle_after_in_out(self, in_out_frontier):
        rect_frontier = RectFromInOutFrontier(in_out_frontier)
        rect_frontier = RectUniqueFrontier(rect_frontier)
        rect_frontier.print_best()
        rect_frontier = PNPrmseFrontier(rect_frontier)
        return rect_frontier

    def get_points(self, cur_frame, last_frame, last_points, predict_matrix):
        self.state.cur_frame = cur_frame
        self.state.last_points = last_points
        self.state.predict_matrix = predict_matrix

        side_frontiers_in, side_frontiers_out = self.line_frontiers()
        if side_frontiers_in is None:
            return None
        in_frontier, out_frontier = self.rectangle_frontiers(side_frontiers_in, side_frontiers_out)
        if in_frontier is None:
            return None
        in_out_frontier = self.in_out_frontier(in_frontier, out_frontier)
        if out_frontier is None:
            return None
        rect_frontier = self.rectangle_after_in_out(in_out_frontier)
        if rect_frontier is None:
            return None
        # show_best(rect_frontier, show_all=True)

        rect_frontier = rect_frontier.top_current()[0]
        lines = [candidate.line for candidate in rect_frontier.lines]
        intersections = screen_lines_to_points(lines)
        return intersections

    tracker_params = TrackerParams()

    def __init__(self, model_vertices, camera_params, video_source, frame_pixels):
        super().__init__(model_vertices, camera_params, video_source, frame_pixels)
