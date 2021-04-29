from functools import partial
from maenv.environment import TeamsEnv


def ma_env(env, **kwargs) -> TeamsEnv:
    return env(**kwargs)


REGISTRY = {"ma": partial(ma_env, env=TeamsEnv)}
