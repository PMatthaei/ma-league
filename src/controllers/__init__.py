from .basic_controller import BasicMAC
from .ensemble_agent_controller import EnsembleInferenceMAC
from .distinct_agents_controller import DistinctMAC
from .sfs_controller import SFSController

REGISTRY = {
    "basic_mac": BasicMAC,
    "distinct_mac": DistinctMAC,
    "combine_mac": EnsembleInferenceMAC,
    "gpe_mac": SFSController
}