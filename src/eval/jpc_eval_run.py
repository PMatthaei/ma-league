from multiprocessing.dummy import Process
from multiprocessing.dummy import Manager

import numpy as np

from eval.methods import avg_proportional_loss
from learners.learner import Learner
from runs.self_play_run import SelfPlayRun


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
        manager = Manager()
        self.policies = manager.list()
        [self.policies.append([None] * self.instances) for _ in range(self.instances)]
        self.jpc_matrix = manager.list()
        [self.jpc_matrix.append([None] * self.instances) for _ in range(self.instances)]

    def run_training(self, instance: int, policies):
        # Start a self play run
        self.args.t_max = 100
        play = SelfPlayRun(args=self.args, logger=self.logger)
        play.start()

        # Save policies for evaluation
        # TODO are these really saved or just references which are changed by another selfplayrun
        policies[instance] = PolicyPair(one=play.home_learner, two=play.opponent_learner)

    def start(self) -> None:
        """
        Evaluate a policy pair with joint policy correlation.
        Therefore the policy is playing against it`s training partner to measure if there is correlation in results.
        """
        procs = []
        # Train policies
        for instance in range(self.instances):
            proc = Process(target=self.run_training, args=(instance, self.policies,))
            proc.start()
            procs.append(proc)

        [proc.join() for proc in procs]

        # Evaluate policies # TODO parallelize, but requires underlying selfplay run to work with multiple callers
        for i in range(self.instances):  # Let all instances play against each other
            for j in range(self.instances):
                self.run_eval((i, j))

        self.stepper.close_env()
        self.logger.console_logger.info("Finished JPC Evaluation")
        jpc_matrix = np.array(self.jpc_matrix) # convert to numpy for calculations
        self.logger.console_logger.info("Avg. Proportional Loss: {}".format(avg_proportional_loss(jpc_matrix)))

    def run_eval(self, instance_pair) -> None:
        """
        Evaluates each learner pairing in sequence on the underlying self play run.
        :param instance_pair:
        :return:
        """
        i, j = instance_pair
        self.logger.console_logger.info(
            "Evaluating player one from instance {} against player two from instance {} for {} episodes"
                .format(i, j, self.eval_episodes)
        )
        # TODO are learners really persisted and the ones trained?
        self.home_learner, self.opponent_learner = self.policies[i].one, self.policies[j].two
        episode = 0
        home_ep_rewards, away_ep_rewards = [], []
        while episode < self.eval_episodes:
            home_batch, away_batch, last_env_info = self.stepper.run()
            home_ep_rewards.append(np.sum(home_batch["reward"].flatten().cpu().numpy()))
            away_ep_rewards.append(np.sum(away_batch["reward"].flatten().cpu().numpy()))
            episode += 1
        self.jpc_matrix[i][j] = np.mean(home_ep_rewards) + np.mean(away_ep_rewards)
