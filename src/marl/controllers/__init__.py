from .basic_controller import BasicMAC
from .ensemble_agent_controller import EnsembleMAC
from .distinct_agents_controller import DistinctMAC
from .sfs_controller import SFSController

REGISTRY = {
    "basic": BasicMAC,
    "distinct": DistinctMAC,
    "ensemble": EnsembleMAC,
    "gpe": SFSController
}