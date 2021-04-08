import unittest

import numpy as np

from eval.joint_policy_correlation import JointPolicyCorrelationEvaluation


class JointPolicyCorrelationTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.jpc = JointPolicyCorrelationEvaluation()
        # Example from paper: A Unified Game Theoretic Approach to RL
        self.jpc.jpc_matrix = np.array([
            [30.7, 30.9, 23.9, 3.7, 9.2],
            [29.9, 30.8, 17.8, 11., 15.5],
            [14.6, 12.9, 30.3, 7.3, 23.8],
            [27.3, 31.7, 27.6, 30.6, 26.2],
            [25.1, 27.3, 29.6, 5.3, 29.8]
        ])
        pass

    def test_avg_proportional_loss(self):
        result = self.jpc.avg_proportional_loss()
        self.assertEqual(0.342, round(result, 3))


