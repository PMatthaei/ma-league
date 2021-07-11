import os

import torch as th


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config


def find_latest_model_path(path: str, load_step: int = 0):
    timesteps = []

    # Go through all files in args.checkpoint_path
    for name in os.listdir(path):
        full_name = os.path.join(path, name)
        # Check if they are dirs the names of which are numbers -> collect
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))
    model_path = os.path.join(path, str(timestep_to_load))
    return model_path, timestep_to_load
