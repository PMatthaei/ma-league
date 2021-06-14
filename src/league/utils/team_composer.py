import itertools

import enum
from collections import Counter

from maenv.core import RoleTypes, UnitAttackTypes

from custom_logging.logger import MainLogger


class Team:
    def __init__(self):
        self.unit_occurrence_dict = dict()

    def difference(self, team):
        t1_counts = Counter(self.unit_occurrence_dict)
        t2_counts = Counter(team)
        diff = [t1_counts[unit] - t2_counts[unit] if unit in t2_counts else t1_counts[unit] for unit in t1_counts]
        in_swaps = [x for x in diff if x > 0]
        weighting = len(in_swaps) / sum(self.unit_occurrence_dict.values())
        dist = sum(in_swaps) * weighting
        return dist


class TeamComposer:
    def __init__(self, *characteristics: [enum.EnumMeta]):
        self.logger = MainLogger.console_logger()
        assert characteristics is not None and len(
            characteristics) > 0, "Please supply characteristics to create units from."
        self.characteristics = characteristics

    def compose_unique_teams(self, team_size=1):
        """
        Build all possible team compositions considering a pool of unique units and a given team size.
        :param team_size:
        :return:
        """
        units = self.compose_unique_units()
        team_comps = list(itertools.combinations_with_replacement(units, team_size))
        self.logger.debug("Created {} unique team compositions with team size {}.".format(len(team_comps), team_size))
        return [self._to_team_build_plan(comp_id, comp) for comp_id, comp in enumerate(team_comps)]

    def compose_unique_units(self):
        """
        Build a pool of unique units defined by the provided characteristics.
        :return:
        """
        self.logger.debug("Composing units with {} characteristics:".format(len(self.characteristics)))
        self.logger.debug((["Name {}: - Size: {}".format(c, len(c)) for c in self.characteristics]))
        units = list(itertools.product(*self.characteristics))
        self.logger.debug("Created {} unique units.".format(len(units)))
        return map(lambda unit: {'role': unit[0], 'attack_type': unit[1]}, units)

    @staticmethod
    def _to_team_build_plan(tid, units, is_scripted=False):
        """
        Convert to a build plan which is used to setup the environment.
        :param tid:
        :param units:
        :param is_scripted:
        :return:
        """
        return {  # Team
            "id": tid,
            "is_scripted": is_scripted,
            "units": units
        }


if __name__ == '__main__':
    n = 3
    teams = TeamComposer(RoleTypes, UnitAttackTypes).compose_unique_teams(n)
    team_hash = hash(str(teams[0]))
    print(teams[0])
    print("Creating build plan for team with hash {}".format(team_hash))
