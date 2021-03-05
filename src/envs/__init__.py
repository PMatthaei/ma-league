import logging
from functools import partial

from multiagent.core import RoleTypes, UnitAttackTypes
from smac.env import StarCraft2Env

from multiagent.environment import MAEnv
from multiagent.scenarios import team
from bin.team_plans_example import TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS

import sys
import os

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


def smac_env(env, **kwargs) -> StarCraft2Env:
    return env(**kwargs)


def ma_env(env, **kwargs) -> MAEnv:
    # load scenario from script
    # TODO: init via kwargs from yaml as with SC2 env
    T = [
        {
            "is_scripted": False,
            "units": [  # Team 1
                {
                    "role": RoleTypes.TANK,
                    "attack_type": UnitAttackTypes.MELEE
                },
                {
                    "role": RoleTypes.TANK,
                    "attack_type": UnitAttackTypes.MELEE
                },
            ]
        },
        {
            "is_scripted": False,
            "units": [  # Team 2
                {
                    "role": RoleTypes.ADC,
                    "attack_type": UnitAttackTypes.RANGED
                },
                {
                    "role": RoleTypes.ADC,
                    "attack_type": UnitAttackTypes.RANGED
                },
            ]
        },
    ]
    scenario = team.load('teams.py').TeamsScenario(T)
    # create world
    world = scenario.make_teams_world(grid_size=10.0)
    # create multi-agent environment
    return env(world=world,
               reset_callback=scenario.reset_world, reward_callback=scenario.reward,
               observation_callback=scenario.observation, info_callback=None, done_callback=scenario.done,
               log=True, log_level=logging.ERROR,
               headless=True)


REGISTRY = {}
REGISTRY["sc2"] = partial(smac_env, env=StarCraft2Env)
REGISTRY["ma"] = partial(ma_env, env=MAEnv)
