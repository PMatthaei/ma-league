import logging
from logging import Logger
from typing import Dict

import numpy as np


class CustomConsoleLogger:
    def __init__(self, console: Logger):
        self.console = console

    @staticmethod
    def console_logger():
        logger = logging.getLogger('ma-league')
        logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    def log(self, stats):
        output = self._format(stats)
        self.console.info(output)

    @staticmethod
    def _format(stats: Dict) -> str:
        """
        Format provided stats into a single string and print into console.
        :return:
        """
        def _skip(collectible: str):
            """
            Decide which elements of the stats dict to skip.
            :param collectible:
            :return:
            """
            return collectible == "episode" or "actions_taken_extract_greedy_actions" in collectible

        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*stats["episode"][-1])
        i = 0
        for (collectible, data) in sorted(stats.items()):
            if _skip(collectible):
                continue
            i += 1
            window = 5 if collectible != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] for x in stats[collectible][-window:]]))
            log_str += "{:<25}{:>8}".format(collectible + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        return log_str
