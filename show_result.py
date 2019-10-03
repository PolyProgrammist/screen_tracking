import subprocess
from utils import TrackingDataReader


def show_result(video_output):
    subprocess.run(['xdg-open', video_output])


if __name__ == "__main__":
    reader = TrackingDataReader()
    show_result(reader.show_input())
