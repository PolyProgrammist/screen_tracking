import subprocess
from utils import TrackingDataReader

reader = TrackingDataReader()

subprocess.run(['xdg-open', reader.show_input()])