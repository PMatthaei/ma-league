import collections
import os
from copy import deepcopy

import yaml
from maenv.utils.enums import as_enum


def set_agents_only(config):
    """
    Ensure every team in the config is NOT a scripted AI but instead a learning agent.
    :param config:
    :return:
    """
    config["env_args"]["match_build_plan"][0]["is_scripted"] = False
    config["env_args"]["match_build_plan"][1]["is_scripted"] = False


def load_match_build_plan(path, env_args):
    """
    Load a match plan defined in the environment config.
    :param path:
    :param env_args:
    :return:
    """
    import json
    with open(f'{os.path.join(path)}/config/teams/{env_args["match_build_plan"]}.json') as f:
        teams_build_plan = json.load(f, object_hook=as_enum)
        env_args["match_build_plan"] = teams_build_plan


def recursive_dict_update(dest_dict, source_dict):
    """
    Update the destination dict with data from the source dict.
    This will override shared keys with values from the source
    :param dest_dict:
    :param source_dict:
    :return:
    """
    for k, v in source_dict.items():
        if isinstance(v, collections.Mapping):
            dest_dict[k] = recursive_dict_update(dest_dict.get(k, {}), v)
        else:
            dest_dict[k] = v
    return dest_dict


def config_copy(config):
    """
    Recursively copy a config.
    :param config:
    :return:
    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def get_default_config(path):
    with open(os.path.join(path, "config", "default.yaml"), "r") as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)


def get_config(params, arg_name, subfolder, path):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(path, "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
