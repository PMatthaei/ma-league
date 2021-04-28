from .parallel_stepper import ParallelStepper
from .episode_stepper import EpisodeStepper
from .self_play_parallel_stepper import SelfPlayParallelStepper
from .self_play_stepper import SelfPlayStepper

REGISTRY = {}
REGISTRY["episode"] = EpisodeStepper
REGISTRY["parallel"] = ParallelStepper

SELF_REGISTRY = {}
SELF_REGISTRY["episode"] = SelfPlayStepper
SELF_REGISTRY["parallel"] = SelfPlayParallelStepper
