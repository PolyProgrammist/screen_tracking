#!/usr/bin/env python3

import coloredlogs
import click

import tracking, compare, write_result, show_result

from utils import TrackingDataReader


@click.command()
@click.option('--test', 'test_directory', default='resources/tests/generated_tv_on', help='Test directory for input and output')
@click.option('--descr', 'test_description', default='test_description.yml', help='Test directory for input and output')
@click.option('--video-output', 'video_output', default='out.mp4', help='Test directory for input and output')
@click.option('--tracking-result', 'tracking_result', default='tracking_result.yml', help='Test directory for input and output')
def test(test_directory, test_description, video_output, tracking_result):
    coloredlogs.install()
    reader = TrackingDataReader(
        test_directory,
        test_description=test_description,
        video_output=video_output,
        tracking_result=tracking_result
    )
    tracking.track(*reader.tracker_input())
    compare.compare(*reader.compare_input())
    write_result.write_result(*reader.draw_input())
    show_result.show_result(reader.show_input())


if __name__ == "__main__":
    test()
