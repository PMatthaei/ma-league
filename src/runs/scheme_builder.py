import torch as th

from marl.components.transforms import OneHot


class SchemeBuilder:

    def __init__(self, args, env_info):
        self.args = args
        self.env_info = env_info
        pass

    def _build_schemes(self):
        scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (self.env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        if self.args.sfs:
            scheme.update({"features": {"vshape": (self.args.sfs_n_features,)}})

        groups = {
            "agents": self.args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }
        return groups, preprocess, scheme