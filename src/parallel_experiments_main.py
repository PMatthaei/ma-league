import argparse
import datetime
import os
import sys
import threading
from os.path import dirname, abspath

import torch as th

from league.processes.experiment_process import ExperimentProcess
from copy import deepcopy

from league.processes.training.ma_experiment_process import MultiAgentExperimentProcess

th.multiprocessing.set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Deactivate message from envs built pygame

if __name__ == '__main__':
    # Handle pre experiment start arguments without sacred
    params = deepcopy(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=3, type=int)
    args, _ = parser.parse_known_args(sys.argv)

    src_dir = dirname(abspath(__file__))

    unique_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{dirname(dirname(abspath(__file__)))}/results/parallel_experiments_{unique_token}'

    procs = []
    # Start multiple experiments
    for idx in range(args.n):
        proc = MultiAgentExperimentProcess(idx=idx, params=params, configs_dir=src_dir, log_dir=log_dir)
        procs.append(proc)

    [r.start() for r in procs]

    # Wait for experiments to finish
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
