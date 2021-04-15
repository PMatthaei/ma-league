from .parallel_stepper import ParallelStepper
from .episode_stepper import EpisodeStepper

REGISTRY = {}
REGISTRY["episode"] = EpisodeStepper
REGISTRY["parallel"] = ParallelStepper
