from time import time

from multiagent.core import RoleTypes, UnitAttackTypes

from league.rl import Trajectory, loss_function
from multiagent.environment import MAEnv

from league.league import League
from league.utils.team_composer import TeamComposer
from runners.league_runner import ActorLoop

LOOPS_PER_ACTOR = 1000
BATCH_SIZE = 512
TRAJECTORY_LENGTH = 64


class Coordinator:
    """Central worker that maintains payoff matrix and assigns new matches."""

    def __init__(self, league):
        self.league = league

    def send_outcome(self, home_player, away_player, outcome):
        self.league.update(home_player, away_player, outcome)
        if home_player.ready_to_checkpoint():
            self.league.add_player(home_player.checkpoint())


class Learner:
    """Learner worker that updates agent parameters based on trajectories."""

    def __init__(self, player):
        self.player = player
        self.trajectories = []
        self.optimizer = AdamOptimizer(learning_rate=3e-5, beta1=0, beta2=0.99,
                                       epsilon=1e-5)

    def get_parameters(self):
        return self.player.agent.get_weights()

    def send_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def update_parameters(self):
        trajectories = self.trajectories[:BATCH_SIZE]
        self.trajectories = self.trajectories[BATCH_SIZE:]
        loss = loss_function(self.player.agent, trajectories)
        self.player.agent.steps += num_steps(trajectories)
        self.player.agent.set_weights(self.optimizer.minimize(loss))

    @background
    def run(self):
        while True:
            if len(self.trajectories) > BATCH_SIZE:
                self.update_parameters()


def main():
    """Trains the league."""
    team_size = 3
    team_compositions = TeamComposer(RoleTypes, UnitAttackTypes).compose_unique_teams(team_size)
    league = League(initial_agents={}) # TODO how to initialize
    coordinator = Coordinator(league)
    learners = []
    actors = []
    for idx in range(league.roles_per_initial_agent() * len(team_compositions)):
        player = league.get_player(idx)
        learner = Learner(player)
        learners.append(learner)
        # Add 16000 loops per initial player
        actors.extend([ActorLoop(player, coordinator) for _ in range(16000)])

    for learner in learners:
        learner.run()
    for actor in actors:
        actor.run()

    # Wait for training to finish.
    join()


if __name__ == '__main__':
    main()
