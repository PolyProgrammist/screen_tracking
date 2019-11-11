#!/usr/bin/env python3

import click
import coloredlogs
import numpy as np

from screen_tracking.common import TrackingDataReader
from screen_tracking.test import draw_result, show_result, compare
from screen_tracking.tracker.hough_heuristics import tracking as hough_tracking


@click.command()
@click.option('--test', 'test_directory', default=TrackingDataReader.DEFAULT_TEST,
              help='Test directory for input and output')
@click.option('--descr', 'test_description', default=TrackingDataReader.DEFAULT_DESCRIPTION_FILE, help='Test description file')
@click.option('--video-output', 'video_output', default=TrackingDataReader.DEFAULT_VIDEO_OUTPUT, help='Tracker video output file')
@click.option('--tracking-result', 'tracking_result_output', default=TrackingDataReader.DEFAULT_TRACKING_OUTPUT,
              help='Tracker result output file')
@click.option('-s', '--steps', multiple=True, default=['tracking', 'compare', 'write_video', 'show_video'],
              help='Steps to execute. Possible steps are: tracking, compare, write_video, show_video')
@click.option('-a', '--algorithm', type=click.Choice(['hough']), default='hough')
def test(steps, algorithm, **kwargs):
    np.random.seed(42)
    coloredlogs.install()
    reader = TrackingDataReader(
        **kwargs
    )
    if 'tracking' in steps:
        if algorithm == 'hough':
            hough_tracking.track(*reader.tracker_input())
    if 'compare' in steps:
        compare.compare(*reader.compare_input())
    if 'write_video' in steps:
        draw_result.write_result(*reader.draw_input())
    if 'show_video' in steps:
        show_result.show_result(reader.show_input())


if __name__ == "__main__":
    test()
