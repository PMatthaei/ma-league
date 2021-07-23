import os
import sys
import threading
from os.path import dirname, abspath

import torch as th

from league.processes.experiment_process import ExperimentProcess
from copy import deepcopy

th.multiprocessing.set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    src_dir = dirname(abspath(__file__))
    procs = []
    # Start league instances
    for _ in range(2):
        proc = ExperimentProcess(params=params, src_dir=src_dir)
        procs.append(proc)

    [r.start() for r in procs]

    # Wait for processes to finish
    [r.join() for r in procs]

    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
