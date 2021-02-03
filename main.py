import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.interfaces.policy import RandomPolicy
from multiagent.scenarios import team


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='team_simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = team.load(args.scenario).TeamSimpleScenario()
    # create world
    world = scenario.make_teams_world(grid_size=10.0)
    # create multi-agent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=None,
                        done_callback=scenario.done,
                        log=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create random policies for each agent in each team
    all_policies = [[RandomPolicy(env, agent.id) for agent in team.members] for team in world.teams]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for tid, team in enumerate(world.teams):
            team_policy = all_policies[tid]
            for aid, agent in enumerate(team.members):
                agent_policy = team_policy[aid]
                act_n.append(agent_policy.action(obs_n[aid]))

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()

        if any(done_n):
            env.reset()
