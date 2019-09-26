#!/usr/bin/env python3

import coloredlogs
import click

from screen_tracking.common.utils import TrackingDataReader
from screen_tracking.tracker import tracking

from screen_tracking.test import draw_result, show_result, compare


@click.command()
@click.option('--test', 'test_directory', default='resources/tests/generated_tv_on',
              help='Test directory for input and output')
@click.option('--descr', 'test_description', default='test_description.yml', help='Test description file')
@click.option('--video-output', 'video_output', default='out.mp4', help='Tracker video output file')
@click.option('--tracking-result', 'tracking_result_output', default='tracking_result.yml',
              help='Tracker result output file')
@click.option('-s', '--steps', multiple=True, default=['tracking', 'compare', 'write_video', 'show_video'],
              help='Steps to execute. Possible steps are: tracking, compare, write_video, show_video')
def test(test_directory, steps, **kwargs):
    coloredlogs.install()
    reader = TrackingDataReader(
        test_directory,
        **kwargs
    )
    if 'tracking' in steps:
        tracking.track(*reader.tracker_input())
    if 'compare' in steps:
        compare.compare(*reader.compare_input())
    if 'write_video' in steps:
        draw_result.write_result(*reader.draw_input())
    if 'show_video' in steps:
        show_result.show_result(reader.show_input())


if __name__ == "__main__":
    test()
