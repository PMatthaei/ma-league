from .basic_controller import BasicMAC
from .distinct_agents_controller import DistinctMAC
from .ensemble_agent_controller import EnsembleMAC
from .sfs_controller import SFSController

REGISTRY = {
    "basic": BasicMAC,
    "distinct": DistinctMAC,
    "ensemble": EnsembleMAC,
    "gpe": SFSController
}