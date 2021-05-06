class MultiAgentController:
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        raise NotImplementedError()

    def forward(self, ep_batch, t, test_mode=False):
        raise NotImplementedError()

    def init_hidden(self, batch_size):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def load_state(self, other_mac):
        raise NotImplementedError()

    def cuda(self):
        raise NotImplementedError()

    def save_models(self, path, name):
        raise NotImplementedError()

    def load_models(self, path, name):
        raise NotImplementedError()
