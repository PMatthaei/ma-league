from runs.run import NormalPlayRun
from steppers.self_play_stepper import SelfPlayStepper

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer


class SelfPlayRun(NormalPlayRun):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.args = args
        self.logger = logger

        # Init runner so we can get env info
        self.stepper: SelfPlayStepper = SelfPlayStepper(args=args, logger=logger)

        # Set up schemes and groups here
        env_info = self.stepper.get_env_info()
        # Calculate per multi-agent number of agents
        self.args.n_agents = int(env_info["n_agents"] / 2)  # TODO: assuming same team size
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        groups, preprocess, scheme = self._build_schemes()

        # Buffers
        buffer_size = self.args.buffer_size
        device = "cpu" if self.args.buffer_cpu_only else self.args.device
        self.opponent_buffer = ReplayBuffer(scheme, groups, buffer_size, env_info["episode_limit"] + 1,
                                            preprocess=preprocess,
                                            device=device)

        self.home_buffer = ReplayBuffer(scheme, groups, buffer_size, env_info["episode_limit"] + 1,
                                        preprocess=preprocess,
                                        device=device)

        # Setup multi-agent controller here
        self.home_mac = mac_REGISTRY[self.args.mac](self.home_buffer.scheme, groups, self.args)
        self.opponent_mac = mac_REGISTRY[self.args.mac](self.opponent_buffer.scheme, groups, self.args)

        # Give runner the scheme
        self.stepper.initialize(scheme=scheme, groups=groups, preprocess=preprocess, home_mac=self.home_mac,
                                opponent_mac=self.opponent_mac)

        # Learners
        self.home_learner = le_REGISTRY[self.args.learner](self.home_mac,
                                                           scheme,
                                                           logger,
                                                           self.args,
                                                           name="home")
        self.opponent_learner = le_REGISTRY[self.args.learner](self.opponent_mac,
                                                               scheme,
                                                               logger,
                                                               self.args,
                                                               name="opponent")

        self.learners = [self.home_learner, self.opponent_learner]

        # Activate CUDA mode if supported
        if self.args.use_cuda:
            self.home_learner.cuda()
            self.opponent_learner.cuda()

    def _train_episode(self, episode_num):
        # Run for a whole episode at a time
        home_batch, opponent_batch = self.stepper.run(test_mode=False)

        self.home_buffer.insert_episode_batch(home_batch)
        self.opponent_buffer.insert_episode_batch(opponent_batch)

        # Sample batch from buffer if possible
        batch_size = self.args.batch_size
        if self.home_buffer.can_sample(batch_size) and self.opponent_buffer.can_sample(batch_size):
            home_sample = self.home_buffer.sample(batch_size)
            opponent_sample = self.opponent_buffer.sample(batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t_h = home_sample.max_t_filled()
            max_ep_t_o = opponent_sample.max_t_filled()
            home_sample = home_sample[:, :max_ep_t_h]
            opponent_sample = opponent_sample[:, :max_ep_t_o]

            device = self.args.device
            if home_sample.device != device:
                home_sample.to(device)

            if opponent_sample.device != device:
                opponent_sample.to(device)

            self.home_learner.train(home_sample, self.stepper.t_env, episode_num)
            self.opponent_learner.train(opponent_sample, self.stepper.t_env, episode_num)