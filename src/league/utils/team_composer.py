from __future__ import annotations

import itertools

import enum
from collections import Counter
from typing import List

from maenv.core import RoleTypes, UnitAttackTypes


class Team:
    def __init__(self, tid: int, units: List, is_scripted=False):
        self.id_ = tid,
        self.units = list(units)
        self.unit_ids = [unit["uid"] for unit in self.units]
        self.is_scripted = is_scripted

    def difference(self, team: Team):
        """
        Calculate the swap-distance for a given team.
        :param team:
        :return:
        """
        t1_counts = Counter(self.unit_ids)
        t2_counts = Counter(team.unit_ids)
        diff = [t1_counts[unit] - t2_counts[unit] if unit in t2_counts else t1_counts[unit] for unit in t1_counts]
        in_swaps = [x for x in diff if x > 0]
        weighting = len(in_swaps) / sum(t1_counts.values())
        dist = sum(in_swaps) * weighting
        return dist


class TeamComposer:
    def __init__(self, *characteristics: [enum.EnumMeta]):
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
        return [self._to_team_build_plan(comp_id, comp) for comp_id, comp in enumerate(team_comps)]

    def compose_unique_units(self):
        """
        Build a pool of unique units defined by the provided characteristics.
        :return:
        """
        units = list(enumerate(list(itertools.product(*self.characteristics))))
        return map(lambda unit: {'uid': unit[0], 'role': unit[1][0], 'attack_type': unit[1][1]}, units)

    @staticmethod
    def to_teams(plans: List[dict]) -> List[Team]:
        return [Team(**plan) for plan in plans]

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
            "tid": tid,
            "is_scripted": is_scripted,
            "units": units
        }


if __name__ == '__main__':
    n = 3
    teams = TeamComposer(RoleTypes, UnitAttackTypes).compose_unique_teams(n)
    team_hash = hash(str(teams[0]))
    print(teams[0])
    print("Creating build plan for team with hash {}".format(team_hash))
