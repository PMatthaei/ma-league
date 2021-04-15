from multiprocessing.dummy import Process
from multiprocessing.dummy.connection import Connection

from gym.vector.utils import CloudpickleWrapper


class EnvWorker(Process):
    def __init__(self, remote: Connection, env: CloudpickleWrapper, policy_team_id: int):
        super().__init__()
        self.remote = remote
        self.env = env
        self.policy_team_id = policy_team_id

        self.away = None
        self.closed = False

    def run(self) -> None:
        # Make environment
        env = self.env.fn()
        # Handle incoming commands from the remote connection within another process
        while not self.closed:
            cmd, data = self.remote.recv()
            if cmd == "step":
                actions = data
                # Take a step in the environment
                obs, reward, done_n, env_info = env.step(actions)
                # Return the observations, avail_actions and state to make the next action
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                self.remote.send({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward[self.policy_team_id],
                    "terminated": done_n[self.policy_team_id],
                    "info": env_info
                })
            elif cmd == "reset":
                env.reset()
                self.remote.send({
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs()
                })
            elif cmd == "close":
                env.close()
                self.remote.close()
                self.closed = True
            elif cmd == "get_env_info":
                self.remote.send(env.get_env_info())
            elif cmd == "get_stats":
                # self.remote.send(env.get_stats())
                self.remote.send({})  # TODO reimplement?
            else:
                raise NotImplementedError
