from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from custom_logging.utils.plots import plot_greedy_actions


class CustomTensorboardLogger:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.n_actions = None
        self.n_agents = None
        # Save logged data over time to show trends and tendency in data
        self._custom_temporal_data = defaultdict(lambda: {})

    def log(self, key, value, t, log_type):
        if log_type == 'scalar':
            self.log_scalar(key, value, t)
        elif log_type == 'image':
            self._custom_temporal_data[key][t] = value
            images = plot_greedy_actions(greedy_actions=self._custom_temporal_data[key], n_actions=self.n_actions, n_agents=self.n_agents)
            for i, image in enumerate(images):
                self.log_plot(f"{key}_agent_{i}", image, t)
        else:
            raise NotImplementedError(f"Type {log_type} is not implemented for logging to Tensorboard.")

    def log_scalar(self, key, value, t):
            self.summary_writer.add_scalar(tag=key, scalar_value=value, global_step=t)

    def log_plot(self, key, value, t):
            self.summary_writer.add_image(tag=key, img_tensor=value, global_step=t)
