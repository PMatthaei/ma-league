import itertools
from multiprocessing import Pool
from typing import Tuple, List, Union

import torch as th

from eval.methods import avg_proportional_loss
from learners.learner import Learner
from runs.self_play_run import SelfPlayRun


class JointPolicyCorrelationEvaluationRun:
    def __init__(self, args, logger, instances_num: int = 1, eval_episodes=100):
        self.args = args
        self.args.t_max = 200
        self.logger = logger
        self.instances_num = instances_num
        self.eval_episodes = eval_episodes
        self.policies: List[Union[Tuple[Learner, Learner], Tuple[None]]] = [tuple()] * self.instances_num
        self.jpc_matrix: th.Tensor = th.zeros([self.instances_num, self.instances_num], dtype=th.float32)

    def start(self) -> None:
        """
        Evaluate a policy pair with joint policy correlation.
        Therefore the policy is playing against it`s training partner to measure if there is correlation in results.
        """
        # Train policies
        results = self.train_instances()
        for index, learners in results:
            self.policies[index] = learners

        # Evaluate policies
        results = self.evaluate_instances()

        for index, value in results:
            self.jpc_matrix[index] = value

        self.logger.console_logger.info("Finished JPC Evaluation")
        self.logger.console_logger.info("Avg. Proportional Loss: {}".format(avg_proportional_loss(self.jpc_matrix)))

    def train_instances(self):
        """
        Let all instances play against each other in parallel fashion
        :return:
        """
        instances = list(range(self.instances_num))
        pool = Pool()
        self.logger.console_logger.info("Train {} instances.".format(len(instances)))
        return pool.map(self.train_instance_pair, instances)

    def train_instance_pair(self, instance: int):
        play = SelfPlayRun(args=self.args, logger=self.logger)
        play.start()
        return instance, (play.home_learner, play.away_learner)

    def evaluate_instances(self) -> List[Tuple[Tuple[int, int], float]]:
        """
        Evaluate each player against each player in every training instance
        :return:
        """
        pairs = list(itertools.product(range(self.instances_num), repeat=2))
        pool = Pool()
        self.logger.console_logger.info("Evaluate {} pairings for {} episodes.".format(len(pairs), self.eval_episodes))
        return pool.map(self.evaluate_instance_pair, pairs)

    def evaluate_instance_pair(self, instance_pair) -> Tuple[Tuple[int, int], float]:
        """
        Evaluates the performance of a instance pairing between player one and two.
        :param instance_pair: A pair of instances to test
        :return:
        """
        i, j = instance_pair
        eval_descriptor = "Eval player 1 from instance {} against player 2 from instance {}".format(i, j)
        self.logger.console_logger.info(eval_descriptor)

        play = SelfPlayRun(args=self.args, logger=self.logger)
        play.set_learners(self.policies[i][0], self.policies[j][1])
        home_mean_r, away_mean_r = play.evaluate_mean_returns(episode_n=self.eval_episodes)
        return instance_pair, (home_mean_r + away_mean_r)
