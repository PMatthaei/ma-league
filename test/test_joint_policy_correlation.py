import unittest

import torch as th

from eval.methods import avg_proportional_loss


class JointPolicyCorrelationTestCases(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_avg_proportional_loss(self):
        jpc = th.tensor([
            [30.7, 30.9, 23.9, 3.7, 9.2],
            [29.9, 30.8, 17.8, 11., 15.5],
            [14.6, 12.9, 30.3, 7.3, 23.8],
            [27.3, 31.7, 27.6, 30.6, 26.2],
            [25.1, 27.3, 29.6, 5.3, 29.8]
        ])
        result = avg_proportional_loss(jpc)
        self.assertEqual(0.342, round(result.item(), 3))


