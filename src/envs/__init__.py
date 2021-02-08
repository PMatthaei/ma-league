from functools import partial
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios import team


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # load scenario from script
    scenario = team.load('team_simple.py').TeamSimpleScenario()
    # create world
    world = scenario.make_teams_world(grid_size=10.0)
    # create multi-agent environment
    return MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=None,
                        done_callback=scenario.done,
                        log=True)
    return env(**kwargs)


REGISTRY = {}
REGISTRY["ma"] = partial(env_fn, env=MultiAgentEnv)
