from time import time

from league.rl import Trajectory, loss_function
from multiagent.environment import MAEnv

from league.league import League

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


class ActorLoop:
    """A single actor loop that generates trajectories.

    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, player, coordinator):
        self.player = player
        self.teacher = get_supervised_agent(player.get_race()) # TODO: how to get teacher?
        self.environment = MAEnv() # TODO: add our env configured with yaml
        self.coordinator = coordinator

    def run(self):
        while True:
            opponent = self.player.get_match()
            trajectory = []
            start_time = time()  # in seconds.
            while time() - start_time < 60 * 60:
                home_observation, away_observation, is_final, z = self.environment.reset()
                student_state = self.player.initial_state()
                opponent_state = opponent.initial_state()
                teacher_state = self.teacher.initial_state()

                while not is_final:
                    student_action, student_logits, student_state = self.player.step(home_observation, student_state)
                    # We mask out the logits of unused action arguments.
                    action_masks = get_mask(student_action)
                    opponent_action, _, _ = opponent.step(away_observation, opponent_state)
                    teacher_logits = self.teacher(observation, student_action, teacher_state)

                    observation, is_final, rewards = self.environment.step(student_action, opponent_action)
                    trajectory.append(Trajectory(
                        observation=home_observation,
                        opponent_observation=away_observation,
                        state=student_state,
                        is_final=is_final,
                        behavior_logits=student_logits,
                        teacher_logits=teacher_logits,
                        masks=action_masks,
                        action=student_action,
                        z=z,
                        reward=rewards,
                    ))

                    if len(trajectory) > TRAJECTORY_LENGTH:
                        trajectory = stack_namedtuple(trajectory)
                        self.learner.send_trajectory(trajectory)
                        trajectory = []
                self.coordinator.send_outcome(student, opponent, self.environment.outcome())


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
    """Trains the AlphaStar league."""
    league = League(initial_agents={}) # TODO how to initialize
    coordinator = Coordinator(league)
    learners = []
    actors = []
    RACES = 3
    INITIAL_AGENTS_PER_RACE = 4
    for idx in range(INITIAL_AGENTS_PER_RACE * RACES):
        player = league.get_player(idx)
        learner = Learner(player)
        learners.append(learner)
        actors.extend([ActorLoop(player, coordinator) for _ in range(16000)])

    for learner in learners:
        learner.run()
    for actor in actors:
        actor.run()

    # Wait for training to finish.
    join()


if __name__ == '__main__':
    main()
