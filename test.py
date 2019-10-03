#!/usr/bin/env python3

import coloredlogs

import tracking
import compare
import write_result
import show_result

from utils import TrackingDataReader

if __name__ == "__main__":
    coloredlogs.install()
    reader = TrackingDataReader()
    tracking.track(*reader.tracker_input())
    compare.compare(*reader.compare_input())
    write_result.write_result(*reader.draw_input())
    show_result.show_result(reader.show_input())
