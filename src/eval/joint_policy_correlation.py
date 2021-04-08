import random
from typing import Tuple, List, Union

import numpy as np

from learners.learner import Learner


class PolicyPair:
    def __init__(self, one: Learner, two: Learner):
        self.one = one
        self.two = two


class JointPolicyCorrelationEvaluation:
    def __init__(self, instances: int = 5, eval_episodes=100):
        self.instances = instances
        self.eval_episodes = eval_episodes
        self.policies: Union[List[PolicyPair], List[None]] = [None] * self.instances
        self.jpc_matrix = np.ones((instances, instances))

    def eval(self) -> np.array:
        """
        Evaluate a policy pair with joint policy correlation.
        Therefore the policy is playing against it`s training partner to measure if there is correlation in results.
        """
        # Train policies
        for i in range(self.instances):
            policy_pair = PolicyPair(one=Learner(), two=Learner())
            # TODO train
            # TODO: save checkpoints from policies
            self.policies[i] = policy_pair

        # Evaluate policies
        for i in range(self.instances):  # Build all policy pairs (diag are policies trained together)
            for j in range(self.instances):
                # TODO: load checkpoints from chosen policies
                policy_one, policy_two = self.policies[i].one, self.policies[j].two
                episode = 0
                returns = []
                # Eval policy pair
                # TODO: Play with loaded learners and generate real returns
                while episode < self.eval_episodes:
                    if i == j:
                        returns.append(random.randint(10, 15))
                    else:
                        returns.append(random.randint(5, 10))
                    episode += 1

                self.jpc_matrix[i, j] = np.mean(returns)

        pass

    def avg_proportional_loss(self):
        diag_mask = np.eye(*self.jpc_matrix.shape, dtype=bool)
        d = np.mean(self.jpc_matrix[diag_mask])
        off_mask = ~diag_mask
        o = np.mean(self.jpc_matrix[off_mask])
        return (d - o) / d


if __name__ == '__main__':
    jpc = JointPolicyCorrelationEvaluation()
    jpc.eval()
    print(jpc.jpc_matrix)
    print(jpc.avg_proportional_loss())
