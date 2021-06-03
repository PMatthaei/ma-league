REGISTRY = {}

from .basic_controller import BasicMAC
from .distinct_agents_controller import DistinctMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["distinct_mac"] = DistinctMAC
