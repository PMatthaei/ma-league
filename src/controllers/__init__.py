from .basic_controller import BasicMAC
from .combined_agents_controller import EnsembleInferenceMAC
from .distinct_agents_controller import DistinctMAC

REGISTRY = {
    "basic_mac": BasicMAC,
    "distinct_mac": DistinctMAC,
    "combine_mac": EnsembleInferenceMAC
}
