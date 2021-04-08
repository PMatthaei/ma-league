REGISTRY = {}

from .episode_stepper import EpisodeStepper
REGISTRY["episode"] = EpisodeStepper

from .parallel_stepper import ParallelStepper
REGISTRY["parallel"] = ParallelStepper