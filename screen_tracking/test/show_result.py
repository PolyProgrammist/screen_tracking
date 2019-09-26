import subprocess


def show_result(video_output):
    subprocess.run(['xdg-open', video_output])
