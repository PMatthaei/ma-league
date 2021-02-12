import logging
from functools import partial

from smac.env import MultiAgentEnv, StarCraft2Env

from multiagent.environment import MAEnv
from multiagent.scenarios import team
from bin.team_plans_example import TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS

import sys
import os


def smac_env(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


def ma_env(env, **kwargs) -> MAEnv:
    # load scenario from script
    # TODO: init via kwargs from yaml
    scenario = team.load('teams.py').TeamsScenario(TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS)
    # create world
    world = scenario.make_teams_world(grid_size=10.0)
    # create multi-agent environment
    return env(world=world,
                 reset_callback=scenario.reset_world,
                 reward_callback=scenario.reward,
                 observation_callback=scenario.observation,
                 info_callback=None,
                 done_callback=scenario.done,
                 log=True, log_level=logging.ERROR)


REGISTRY = {}
REGISTRY["sc2"] = partial(smac_env, env=StarCraft2Env)
REGISTRY["ma"] = partial(ma_env, env=MAEnv)
