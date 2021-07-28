import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule


class Selector:
    def select(self, agent_inputs, avail_actions, t_env, test_mode=False):
        raise NotImplementedError()


class MultinomialActionSelector(Selector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select(self, agent_outputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_outputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions, None


class EpsilonGreedyActionSelector(Selector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select(self, agent_outputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only -> Exploit
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_outputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # unavailable should never be selected!

        random_numbers = th.rand_like(agent_outputs[:, :, 0])  # Per agent random value between 0 and 1
        pick_random = (random_numbers < self.epsilon).long()  # Per agent epsilon dependent action selection
        random_actions = Categorical(avail_actions.float()).sample().long()  # Per agent pick random action
        # Per agent: Choose random action if pick random = 1 if pick random = 0 use Q-value
        pick_greedy = (1 - pick_random)
        picked_actions = pick_random * random_actions + pick_greedy * masked_q_values.max(dim=2)[1]
        return picked_actions, pick_greedy


REGISTRY = {
    "multinomial": MultinomialActionSelector,
    "epsilon_greedy": EpsilonGreedyActionSelector
}
