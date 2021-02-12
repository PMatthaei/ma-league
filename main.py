import argparse
import io
import os
import time

from multiagent.environment import MultiAgentEnv
from multiagent.interfaces.policy import RandomPolicy
from multiagent.scenarios import team

from mlflow import log_metric
import numpy as np

import tensorflow as tf
from bin.team_plans_example import TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS
if __name__ == '__main__':
    writer = tf.summary.create_file_writer(logdir="src/mlruns/tf/logs/{}".format(int(time.time())))
    writer.set_as_default()
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='teams.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = team.load(args.scenario).TeamsScenario(TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS)
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

    episodal_acc_reward = 0
    episodal_rewards_at_t = [list() for x in range(env.max_steps)]
    episode = 1
    total_steps = 0
    total_rewards = []

    taken_actions = [0] * 7

    while total_steps < 10000:
        # query for action from each agent's policy
        act_n = []
        for tid, team in enumerate(world.teams):
            team_policy = all_policies[tid]
            for aid, agent in enumerate(team.members):
                agent_policy = team_policy[aid]
                act_n.append(agent_policy.action(obs_n[aid]))

        counts = [act_n.count(i) for i in np.arange(7)]
        taken_actions = [current + count for current, count in zip(taken_actions, counts)]

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # calc and log reward
        episodal_rewards_at_t[env.t - 1].append(reward_n[0])
        total_rewards.append(reward_n[0])
        episodal_acc_reward += reward_n[0]
        total_steps += 1

        log_metric(key="episode_avg_reward", value=np.average(episodal_rewards_at_t[env.t - 1]), step=env.t - 1)
        tf.summary.scalar("episode_avg_reward", data=np.average(episodal_rewards_at_t[env.t - 1]), step=env.episode)
        tf.summary.scalar("avg_reward", data=np.average(total_rewards), step=total_steps)
        tf.summary.scalar("max_reward", data=np.max(total_rewards), step=total_steps)
        tf.summary.scalar("min_reward", data=np.min(total_rewards), step=total_steps)

        # render all agent views
        env.render()

        if any(done_n):
            env.reset()
            episodal_acc_reward = 0
            episode += 1
