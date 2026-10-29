import datetime
import os
import sys

import numpy as np


class Logger(object):
    """Writes the same output to several file streams, flushing after every write."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        pass


def create_logger(directory="", name="logs.txt"):
    """Duplicates stdout and stderr into `directory + name`."""
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)
    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """Creates and returns a new run folder: log_path/<date of the day>/<exp_id>/, exp_id one more than the last."""
    log_today = f"{log_path}{datetime.date.today()}/"
    os.makedirs(log_today, exist_ok=True)
    exps = [int(f) for f in os.listdir(log_today) if f.isdigit()]
    log_folder = log_today + f"{np.max(exps) + 1 if len(exps) else 0}/"
    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)
    return log_folder
