from copy import deepcopy
from typing import Dict

from nestargs.parser import NestedNamespace

from utils.main_utils import get_default_config, get_config, get_match_build_plan, recursive_dict_update, \
    build_config_argsparser
from utils.run_utils import args_sanity_check


class ConfigBuilder:

    def __init__(self, worker_args, src_dir, log_dir, params):
        self.worker_args = worker_args
        self._src_dir = src_dir
        self._log_dir = log_dir
        self._params = params
        pass

    def build(self, training_idx: int) -> Dict:
        params = deepcopy(self._params)  # Copy to prevent changing for subsequent build() calls

        experiment_config = self._read_config_yamls(params)  # Read yamls in src/config

        experiment_config["log_dir"] = self._log_dir  # set logging directory for instance metrics and model

        if "match_build_plan" in experiment_config['env_args']:  # Load build plan if configured
            plan = experiment_config['env_args']["match_build_plan"]
            experiment_config['env_args']["match_build_plan"] = get_match_build_plan(self._src_dir, plan)

        # Overwrite .yaml config with command params via update
        self._overparse_params(experiment_config, params)

        self._assign_device(experiment_config, training_idx)

        return experiment_config

    def _read_config_yamls(self, params):
        # Get the defaults from default.yaml
        config_dict = get_default_config(self._src_dir)
        # Load env base config
        env_config = get_config(params, "--env-config", "envs", path=self._src_dir)
        league_config = get_config(params, "--league-config", "leagues", path=self._src_dir)

        # Load algorithm base config
        alg_config = get_config(params, "--config", "algs", path=self._src_dir)
        # Integrate loaded dicts into main dict
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, league_config)  # League overwrites env
        experiment_config = recursive_dict_update(config_dict, alg_config)  # Algorithm overwrites all config
        experiment_config = args_sanity_check(experiment_config)  # check args are valid
        return experiment_config

    def _overparse_params(self, experiment_config, params):
        # Build parser to parse the
        parser = build_config_argsparser(experiment_config)
        args, _ = parser.parse_known_args(params)
        args_dict = self._namespace_to_dict(args)
        experiment_config.update(args_dict)

    def _assign_device(self, experiment_config, training_idx):
        if self.worker_args.balance_cuda_workload:
            # Distribute instance workload evenly in round robin manner to all cuda devices
            device_id = training_idx % len(self.worker_args.cuda_devices)
        else:
            device_id = 0
        experiment_config["device"] = self.worker_args.cuda_devices[device_id] if experiment_config[
            "use_cuda"] else "cpu"  # set device depending on cuda

    def _namespace_to_dict(self, args: NestedNamespace):
        args_dict = vars(args)
        for entry in args_dict:
            if isinstance(args_dict[entry], NestedNamespace):
                args_dict[entry] = self._namespace_to_dict(args_dict[entry])  # Recursive convert
        return args_dict
