from copy import deepcopy
from typing import Dict

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
        # Build parser to parse all values known in current config
        parser = build_config_argsparser(experiment_config, params)
        args, _ = parser.parse_known_args(params)
        experiment_config.update(vars(args))
        return experiment_config
