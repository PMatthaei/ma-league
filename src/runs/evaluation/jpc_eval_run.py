import itertools
from torch.multiprocessing import Pool
from typing import Tuple, List, Union

import torch as th

from eval.methods import avg_proportional_loss
from learners.learner import Learner
from runs.train.sp_ma_experiment import SelfPlayMultiAgentExperiment


class JointPolicyCorrelationEvaluationRun:
    def __init__(self, args, logger, instances_num: int = 2, eval_episodes=100):
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
        checkpoints = self.train_instances()

        # Evaluate policies
        indices, values = self.evaluate_instances(checkpoints)

        # Fill JPC matrix with results
        self.jpc_matrix[indices[:, 0], indices[:, 1]] = values

        self.logger.info("Finished JPC Evaluation")
        self.logger.info("Avg. Proportional Loss: {}".format(avg_proportional_loss(self.jpc_matrix)))

    def train_instances(self):
        """
        Let all instances play against each other in parallel fashion.
        :return: checkpoints produced by all workers
        """
        instances = list(range(self.instances_num))
        self.logger.info("Train {} instances.".format(len(instances)))
        with Pool() as pool:
            results = pool.map(self.train_instance_pair, instances)
        return results

    def train_instance_pair(self, instance: int):
        """
        Train agent in Self-Play and save learners as .th files
        since transferring the learner torch construct between processes is complicated.
        :param instance: identifier of the train instance
        :return: the instance id and the path to the checkpointed policies
        """
        play = SelfPlayMultiAgentExperiment(args=self.args, logger=self.logger)
        play.start()
        path = play.save_models(identifier=f"instance_{instance}")
        return path

    def evaluate_instances(self, checkpoints) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate each player against each other.
        :return:
        """
        pairs = list(itertools.product(range(self.instances_num), repeat=2))
        data = list(zip(pairs, [checkpoints] * len(pairs)))
        self.logger.info("Evaluate {} pairings for {} episodes.".format(len(pairs), self.eval_episodes))
        with Pool() as pool:
            results = pool.starmap(self.evaluate_instance_pair, data)
        indices, values = map(list, zip(*results))
        return th.as_tensor(indices), th.as_tensor(values)

    def evaluate_instance_pair(self, instance_pair, checkpoints) -> Tuple[Tuple[int, int], float]:
        """
        Evaluates the performance of a instance pairing between player one and two of the given instances.
        :param instance_pair: pair of instances to test
        :return:
        """
        i, j = instance_pair
        eval_descriptor = "Eval home player from instance {} against away player from instance {}".format(i, j)
        self.logger.info(eval_descriptor)
        play = SelfPlayMultiAgentExperiment(args=self.args, logger=self.logger)
        play.home_learner.load_models(checkpoints[i])
        play.away_learner.load_models(checkpoints[j])
        # Calculate mean return for both policies
        home_mean_r, away_mean_r = play.evaluate_mean_returns(episode_n=self.eval_episodes)
        return instance_pair, (home_mean_r + away_mean_r)
