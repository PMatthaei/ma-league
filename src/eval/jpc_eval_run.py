import itertools
from multiprocessing.dummy import Process, Pool
from multiprocessing.dummy import Manager

import torch as th

from eval.methods import avg_proportional_loss
from learners.learner import Learner
from runs.self_play_run import SelfPlayRun


class TrainInstance(Process):
    def __init__(self, args, logger, instance: int, policies):
        """
        A train instance under JPC performs self play and saves the resulting policies/learners in the policy collection
        :param args:
        :param logger:
        :param instance:
        :param policies:
        """
        super().__init__()
        self.args = args
        self.logger = logger
        self.instance = instance
        self.policies = policies

    def run(self) -> None:
        # Start a self play run
        self.args.t_max = 100
        play = SelfPlayRun(args=self.args, logger=self.logger)
        play.start()

        # Save policy pair for evaluation
        # TODO are these really saved or just references which are changed by another selfplayrun
        self.policies[self.instance] = PolicyPair(one=play.home_learner, two=play.away_learner)


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
        self.child_run_args = args
        self.child_run_args.runner = "episode"
        self.instances = instances
        self.eval_episodes = eval_episodes
        manager = Manager()
        self.policies = manager.list([None] * self.instances)
        self.jpc_matrix = manager.list([[None] * self.instances for _ in range(self.instances)])

    def start(self) -> None:
        """
        Evaluate a policy pair with joint policy correlation.
        Therefore the policy is playing against it`s training partner to measure if there is correlation in results.
        """
        self._init_stepper()
        procs = []
        # Train policies
        for instance in range(self.instances):
            proc = TrainInstance(args=self.child_run_args, logger=self.logger, instance=instance,
                                 policies=self.policies)
            proc.start()
            procs.append(proc)

        [proc.join() for proc in procs]

        # Evaluate policies
        self.run_evals_parallel()

        self.stepper.close_env()
        self.logger.console_logger.info("Finished JPC Evaluation")
        jpc_matrix = th.tensor(self.jpc_matrix)  # convert to numpy for calculations
        self.logger.console_logger.info("Avg. Proportional Loss: {}".format(avg_proportional_loss(jpc_matrix)))

    def run_evals_parallel(self) -> None:
        """
        Let all instances play against each other in parallel fashion
        :return:
        """
        pairs = list(itertools.product(range(self.instances), repeat=2))
        pool = Pool()
        self.logger.console_logger.info(
            "Evaluating {} pairings for {} episodes.".format(len(pairs), self.eval_episodes))
        pool.map(self.run_eval, pairs)

    def run_eval(self, instance_pair) -> None:
        """
        Evaluates the performance of a instance pairing between player one and two.
        :param instance_pair: A pair of instances to test
        :return:
        """
        i, j = instance_pair
        eval_descriptor = "Eval player 1 from instance {} against player 2 from instance {}".format(i, j)
        self.logger.console_logger.info(eval_descriptor)

        # TODO are learners really persisted and the ones trained?
        play = SelfPlayRun(args=self.args, logger=self.logger)
        play.set_learners(self.policies[i].one, self.policies[j].two)
        home_mean_r, away_mean_r = play.evaluate_mean_returns(episode_n=self.eval_episodes)
        self.jpc_matrix[i][j] = home_mean_r + away_mean_r
