import subprocess
from screen_tracking.common.utils import TrackingDataReader


def show_result(video_output):
    subprocess.run(['xdg-open', video_output])
