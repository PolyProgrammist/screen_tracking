#!/usr/bin/env python3

import click
import coloredlogs
import numpy as np
from time import time
import random

from screen_tracking.common import TrackingDataReader
from screen_tracking.test import draw_result, show_result, compare
from screen_tracking.test.compare import compare_one
from screen_tracking.tracker.common_tracker import common_track
from screen_tracking.tracker.hough_heuristics import tracking as hough_tracking
from screen_tracking.tracker.hough_heuristics.tracking import HoughTracker
from screen_tracking.tracker.lukas_kanade import lukas_kanade as lukas_kanade
from screen_tracking.tracker.lukas_kanade.lukas_kanade import LukasKanadeTracker
from screen_tracking.tracker.vadim_farutin.rapid import RapidTracker
from screen_tracking.tracker.vadim_farutin.sift import SiftTracker


def run_tracking(algorithm, kwargs, reader, steps):
    if 'tracking' in steps:
        print('warning: Tracking')
        start = time()
        if algorithm == 'hough':
            common_track(*reader.tracker_input(), HoughTracker)
        elif algorithm == 'lk':
            common_track(*reader.tracker_input(), LukasKanadeTracker)
        elif algorithm == 'rapid':
            common_track(*reader.tracker_input(), RapidTracker)
        elif algorithm == 'sift':
            common_track(*reader.tracker_input(), SiftTracker)
        stop = time()
        print("Time elapsed: " + str(stop - start))

def run_several(kwargs, algorithms, steps, function):
    for algorithm in algorithms:
        suffixes = ['']
        test_count_number = kwargs['test_count_number']
        test_start_number = kwargs['test_start_number']
        if test_count_number != 1 or test_start_number != 0:
            suffixes = [str(i) for i in range(test_start_number, test_start_number + test_count_number)]
        for suffix in suffixes:
            test_dir = 'resources/tests/'
            tests = \
                ['generated_tv_on' + str(i) + '/' for i in range(10)] + \
                ['generated_tv_off' + str(i) + '/' for i in range(10)]
            for test in tests:
                current_test = test_dir + test
                kwargs['tracking_result_output'] = 'tracking_result_' + algorithm + suffix + '.yml'
                kwargs['test_directory'] = current_test
                reader = TrackingDataReader(
                    **kwargs
                )
                function(algorithm, kwargs, reader, steps)


@click.command()
@click.option('--test', 'test_directory', default=TrackingDataReader.DEFAULT_TEST,
              help='Test directory for input and output')
@click.option('--descr', 'test_description', default=TrackingDataReader.DEFAULT_DESCRIPTION_FILE, help='Test description file')
@click.option('--video-output', 'video_output', default=TrackingDataReader.DEFAULT_VIDEO_OUTPUT, help='Tracker video output file')
@click.option('--tracking-result', 'tracking_result_output', default=TrackingDataReader.DEFAULT_TRACKING_OUTPUT,
              help='Tracker result output file')
@click.option('-s', '--steps', multiple=True, default=['tracking', 'compare', 'write_video', 'show_video'],
              help='Steps to execute. Possible steps are: tracking, compare, write_video, show_video')
@click.option('-a', '--algorithm', type=click.Choice(['hough', 'lk', 'rapid', 'sift']), default='hough')
@click.option('--several', 'several', default=False)
@click.option('--test_start_number', 'test_start_number', default=0)
@click.option('--test_count_number', 'test_count_number', default=1)
def test(steps, algorithm, **kwargs):
    np.random.seed(42)
    random.seed(42)

    coloredlogs.install()

    if kwargs['several'] is False:
        reader = TrackingDataReader(
            **kwargs
        )
        run_tracking(algorithm, kwargs, reader, steps)

        if 'compare' in steps:
            compare.compare(*reader.compare_input())
        if 'write_video' in steps:
            draw_result.write_result(*reader.draw_input())
        if 'show_video' in steps:
            show_result.show_result(reader.show_input())
    else:
        run_several(kwargs, [algorithm], steps, run_tracking)
        run_several(kwargs, ['hough', 'rapid'], steps, compare_one)



if __name__ == "__main__":
    test()
