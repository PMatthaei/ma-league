from __future__ import annotations

from enum import Enum

import numpy as np

from custom_logging.utils.preprocessing import extract_greedy_actions, percentage


class Collectibles(Enum):
    """
    Resembles a metric or value collected during an episode or at the end.
    """
    RETURN = {  # Collected during episode
                 "collection_type": list,
                 "preprocessing": [np.mean, np.std],
                 "log_type": "scalar"
             },
    ACTIONS_TAKEN = {  # Collected during episode
                        "collection_type": list,
                        "preprocessing": [extract_greedy_actions],
                        "log_type": "image"
                    },
    WON = {  # Collected at episode end
              "collection_type": list,
              "preprocessing": [percentage],
              "log_type": "scalar"
          },
    DRAW = {  # Collected at episode end
               "is_global": True,
               "collection_type": list,
               "preprocessing": [percentage],
               "log_type": "scalar"
           },
    EPISODE = {  # Collected at episode end
                  "is_global": True,
                  "collection_type": list,
                  "preprocessing": [np.mean, len],
                  "log_type": "scalar"
              },

    @property
    def is_global(self):
        return self.value[0]["is_global"] if "is_global" in self.value[0] else False

    @property
    def collection_type(self):
        return self.value[0]["collection_type"]

    @property
    def precollect(self):
        return self.value[0]["precollect"] if "precollect" in self.value[0] else None

    @property
    def log_type(self):
        return self.value[0]["log_type"]

    @property
    def preprocessing(self):
        return self.value[0]["preprocessing"]

    @property
    def keys(self):
        return [f"{str(self.name).lower()}_{p.__name__}" for p in self.preprocessing]

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
