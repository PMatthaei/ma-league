from copy import deepcopy
from typing import Dict

from nestargs.parser import NestedNamespace

from utils.main_utils import get_default_config, get_config, get_match_build_plan, recursive_dict_update, \
    build_config_argsparser
from utils.run_utils import args_sanity_check


class ConfigBuilder:

    def __init__(self, src_dir, log_dir, params):
        self._src_dir = src_dir
        self._log_dir = log_dir
        self._params = params
        pass

    def build(self, training_idx: int) -> Dict:
        # Get the defaults from default.yaml
        config_dict = get_default_config(self._src_dir)

        # Load env base config
        params = deepcopy(self._params) # Copy to prevent changing for subsequent build() calls
        env_config = get_config(params, "--env-config", "envs", path=self._src_dir)

        league_config = get_config(params, "--league-config", "leagues", path=self._src_dir)

        # Load build plan if configured
        env_args = env_config['env_args']
        if "match_build_plan" in env_args:
            env_args["match_build_plan"] = get_match_build_plan(self._src_dir, env_args)

        # Load algorithm base config
        alg_config = get_config(params, "--config", "algs", path=self._src_dir)

        # Integrate loaded dicts into main dict
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, league_config)  # League overwrites env
        experiment_config = recursive_dict_update(config_dict, alg_config)  # Algorithm overwrites all config
        experiment_config = args_sanity_check(experiment_config)  # check args are valid
        experiment_config["device"] = "cuda" if experiment_config["use_cuda"] else "cpu"  # set device depending on cuda
        experiment_config["log_dir"] = self._log_dir  # set logging directory for instance metrics and model

        # Overwrite .yaml config with command params via update
        # Build parser to parse the
        parser = build_config_argsparser(experiment_config)
        args, _ = parser.parse_known_args(params)
        args_dict = self._namespace_to_dict(args)
        experiment_config.update(args_dict)
        return experiment_config

    def _namespace_to_dict(self, args: NestedNamespace):
        args_dict = vars(args)
        for entry in args_dict:
            if isinstance(args_dict[entry], NestedNamespace):
                args_dict[entry] = self._namespace_to_dict(args_dict[entry]) # Recursive convert
        return args_dict
