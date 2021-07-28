from copy import deepcopy
from typing import Dict, List

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

    def build(self, training_idx: int) -> Dict:
        params = deepcopy(self._params)  # Copy to prevent changing for subsequent build() calls

        config = self._read_config_yamls(params)

        config["log_dir"] = self._log_dir  # set logging directory for instance metrics and model

        if "match_build_plan" in config['env_args']:  # Load build plan if configured
            plan = config['env_args']["match_build_plan"]
            config['env_args']["match_build_plan"] = get_match_build_plan(self._src_dir, plan)

        # Overwrite .yaml config with command params via update
        self._overparse_params(config, params)

        self._assign_device(config, training_idx)

        return config

    def _read_config_yamls(self, params):
        # Get the defaults from default.yaml
        # Load yamls
        config: Dict = get_default_config(self._src_dir)
        env_config = get_config(params, "--env-config", "envs", path=self._src_dir)
        league_config = get_config(params, "--league-config", "leagues", path=self._src_dir)
        alg_config = get_config(params, "--config", "algs", path=self._src_dir)

        config = recursive_dict_update(config, env_config)
        config = recursive_dict_update(config, league_config)  # league overwrites env config
        experiment_config = recursive_dict_update(config, alg_config)  # algorithm overwrites all config
        experiment_config = args_sanity_check(experiment_config)  # check args are valid
        return experiment_config

    def _overparse_params(self, config: Dict, params: List[str]):
        parser = build_config_argsparser(config)  # Build parser to parse the params
        args, _ = parser.parse_known_args(params)
        args_dict = self._namespace_to_dict(args)
        config.update(args_dict)  # Overwrite current config

    def _assign_device(self, config, training_idx):
        if self.worker_args.balance_cuda_workload:
            device_id = training_idx % len(self.worker_args.cuda_devices) # Assign device in round robin manner
        else:
            device_id = 0
        config["device"] = self.worker_args.cuda_devices[device_id] if config["use_cuda"] else "cpu"

    def _namespace_to_dict(self, args: NestedNamespace):
        args_dict = vars(args)
        for entry in args_dict:
            if isinstance(args_dict[entry], NestedNamespace):
                args_dict[entry] = self._namespace_to_dict(args_dict[entry])  # Recursive convert
        return args_dict
