from functools import partial

import sys
import os

from multiagent.environment import TeamsEnv


def ma_env(env, **kwargs) -> TeamsEnv:
    return env(**kwargs)


REGISTRY = {"ma": partial(ma_env, env=TeamsEnv)}
