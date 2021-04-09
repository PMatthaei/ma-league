import random
from typing import List, Union

import numpy as np

from eval.methods import avg_proportional_loss
from learners.learner import Learner
from runs.self_play_run import SelfPlayRun
from steppers.self_play_stepper import SelfPlayStepper


class PolicyPair:
    def __init__(self, one: Learner, two: Learner):
        """
        Represents a pair of policies which learned together in training.
        :param one:
        :param two:
        """
        self.one = one
        self.two = two


class JointPolicyCorrelationEvaluationRun(SelfPlayRun):
    def __init__(self, args, logger, instances: int = 2, eval_episodes=100):
        super().__init__(args, logger)
        self.args = args
        self.logger = logger
        self.instances = instances
        self.eval_episodes = eval_episodes
        self.policies: Union[List[PolicyPair], List[None]] = [None] * self.instances

    def start(self) -> None:
        """
        Evaluate a policy pair with joint policy correlation.
        Therefore the policy is playing against it`s training partner to measure if there is correlation in results.
        """
        # Train policies
        for i in range(self.instances):
            # Start a self play run
            self.args.t_max = 100
            play = SelfPlayRun(args=self.args, logger=self.logger)
            play.start()

            # Save policies for evaluation
            self.policies[i] = PolicyPair(one=play.home_learner, two=play.opponent_learner)

        jpc_matrix = np.ones((self.instances, self.instances))

        # Evaluate policies
        for i in range(self.instances):  # Let all instances play against each other
            for j in range(self.instances):
                self.logger.console_logger.info(
                    "Evaluating player one from instance {} against player two from instance {} for {} episodes"
                        .format(i, j, self.eval_episodes)
                )

                self.home_learner, self.opponent_learner = self.policies[i].one, self.policies[j].two
                episode = 0
                home_ep_rewards, away_ep_rewards = [], []
                while episode < self.eval_episodes:
                    home_batch, away_batch, last_env_info = self.stepper.run()
                    home_ep_rewards.append(np.sum(home_batch["reward"].flatten().cpu().numpy()))
                    away_ep_rewards.append(np.sum(away_batch["reward"].flatten().cpu().numpy()))
                    episode += 1

                jpc_matrix[i, j] = np.mean(home_ep_rewards) + np.mean(away_ep_rewards)

        self.stepper.close_env()
        self.logger.console_logger.info("Finished JPC Evaluation")

        self.logger.console_logger.info("Avg. Proportional Loss: {}".format(avg_proportional_loss(jpc_matrix)))
