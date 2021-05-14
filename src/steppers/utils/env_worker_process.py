from maenv.environment import MAEnv
from torch.multiprocessing import Process, Queue


class EnvWorker(Process):
    def __init__(self, in_q: Queue, out_q: Queue, env: MAEnv):
        """
        Interacts with environment if requested and communicates results back to parent connection.
        :param remote:
        :param env:
        """
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.env = env

    def run(self) -> None:
        # Make environment
        # Handle incoming commands from the remote connection within another process
        while True:
            cmd, data = self.in_q.get()
            if cmd == "step":
                actions = data
                # Take a step in the environment
                obs, reward, done_n, env_info = self.env.step(actions)
                # Return the observations, avail_actions and state to make the next action
                state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                self.out_q.put({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": done_n,
                    "info": env_info
                })
                del actions
            elif cmd == "reset":
                self.env.reset()
                self.out_q.put({
                    "state": self.env.get_state(),
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": self.env.get_obs()
                })
            elif cmd == "close":
                self.env.close()
                self.out_q.close()
                self.in_q.close()
                break
            elif cmd == "get_env_info":
                self.out_q.put(self.env.get_env_info())
            else:
                raise NotImplementedError(f"Unknown message received in environment worker: {cmd}")
