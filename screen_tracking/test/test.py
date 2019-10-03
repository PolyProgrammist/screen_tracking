#!/usr/bin/env python3

import coloredlogs
import click

from screen_tracking.tracker import tracking
from screen_tracking.test import draw_result, show_result, compare

from screen_tracking.common.utils import TrackingDataReader


@click.command()
@click.option('--test', 'test_directory', default='resources/tests/generated_tv_on',
              help='Test directory for input and output')
@click.option('--descr', 'test_description', default='test_description.yml', help='Test description file')
@click.option('--video-output', 'video_output', default='out.mp4', help='Tracker video output file')
@click.option('--tracking-result', 'tracking_result_output', default='tracking_result.yml', help='Tracker result output file')
def test(test_directory, **kwargs):
    coloredlogs.install()
    reader = TrackingDataReader(
        test_directory,
        **kwargs
    )
    tracking.track(*reader.tracker_input())
    compare.compare(*reader.compare_input())
    draw_result.write_result(*reader.draw_input())
    # show_result.show_result(reader.show_input())


if __name__ == "__main__":
    test()
