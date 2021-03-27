from functools import partial

from smac.env import StarCraft2Env

from multiagent.environment import TeamsEnv

import sys
import os

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


def smac_env(env, **kwargs) -> StarCraft2Env:
    return env(**kwargs)


def ma_env(env, **kwargs) -> TeamsEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(smac_env, env=StarCraft2Env)
REGISTRY["ma"] = partial(ma_env, env=TeamsEnv)
