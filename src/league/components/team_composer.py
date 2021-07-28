from __future__ import annotations

import enum
import itertools
import math
from collections import Counter
from functools import reduce
from random import sample
from typing import List, Dict, Union, Tuple

import numpy as np
from maenv.core import RoleTypes, UnitAttackTypes


class Team:
    def __init__(self, tid: int, units: Tuple[Dict], is_scripted: bool = False):
        self.id_ = tid
        self.units: List[Dict] = list(units)
        self._uids: List[int] = [unit["uid"] for unit in self.units]  # IDs of units
        self._rids: List[int] = [unit["role"].value["id"] for unit in self.units]  # IDs of roles
        self.is_scripted: bool = is_scripted

    def get_team_ids(self, query_ids: List[int]):
        """
        :param query_ids: Unit IDs to search for in the team
        :return: Team internal IDs where the corresponding unit can be found
        """
        uids = np.array(self._uids)
        mask = reduce((lambda x, y: x | y), [(uids == uid) for uid in query_ids])
        return np.where(mask)

    def contains(self, unit_ids: Union[List[int], int], unique=False):
        """
        :param unit_ids: Unit IDs of units that should be in the team
        :param unique: Only one unit may have one of the IDs
        :return: True if the team contains (one or more) the units False otherwise
        """
        if isinstance(unit_ids, int):
            unit_ids = [unit_ids]  # transform to list
        is_unit = [uid in unit_ids for uid in self._uids]  # check which units fulfill condition
        if unique:
            i = iter(is_unit)
            return any(i) and not any(i)  # make sure just one unit fulfills condition
        return any(is_unit)  # at least one unit must be of one of the desired types

    @property
    def roles(self, unique=True):
        roles = [unit['role'] for unit in self.units]
        if unique:
            return set(roles)
        return roles

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        if other == None:
            return False
        return self.id_ == other.id_

    def __str__(self):
        return f"Team #{self.id_}"

    def difference(self, team: Team):
        """
        Calculate the swap-distance for a given team.
        :param team:
        :return:
        """
        t1_counts = Counter(self._uids)
        t2_counts = Counter(team._uids)
        diff = [t1_counts[unit] - t2_counts[unit] if unit in t2_counts else t1_counts[unit] for unit in t1_counts]
        in_swaps = [x for x in diff if x > 0]
        weighting = len(in_swaps) / sum(t1_counts.values())
        dist = sum(in_swaps) * weighting
        return dist


class TeamComposer:
    def __init__(self, team_size, characteristics: [enum.EnumMeta]):
        assert characteristics is not None and len(
            characteristics) > 0, "Please supply characteristics to create units from."
        self.characteristics = characteristics
        self.teams: List[Team] = []
        self._compose_unique_teams(team_size)

    def __getitem__(self, item):
        return self.teams[item]

    def get_uids(self, type: enum.Enum, capability: str):
        """
        :param type:
        :param capability:
        :return: All unit ids that poses the given type in the given capability
        """
        assert capability in ["role", "attack_type"], "Unknown capability"
        uids = [unit["uid"] for unit in self.units if unit[capability] == type]
        return uids

    def get_unique_uid(self, role_type: RoleTypes, attack_type: UnitAttackTypes):
        """
        :param role_type:
        :param attack_type:
        :return: UID of the unique unit with the given role and attack type
        """
        assert role_type is not None and attack_type is not None, "Please supply all characteristics to search unit ids."
        uid = [unit["uid"] for unit in self.units if unit["role"] == role_type and unit["attack_type"] == attack_type]
        if len(uid) > 1:
            raise Exception("Consistency error. Described unit is not unique.")
        else:
            return uid[0]

    def sample(self, k, contains=None, unique=False) -> List[Team]:
        """
        :param unique: if the contain argument should be enforced to be unique
        :param k: number of samples to pick
        :param contains: Condition for a sampled team. Has to contain the given units.
        :return: Set of teams
        """
        return sample([team for team in self.teams if contains is None or team.contains(contains, unique)], k=k)

    def _compose_unique_teams(self, team_size):
        """
        Build all possible team compositions considering a pool of unique units and a given team size.
        :param team_size:
        :return:
        """
        units = list(self._compose_unique_units())
        self.units = units
        team_comps = list(itertools.combinations_with_replacement(units, team_size))
        build_plans = [self._to_team_build_plan(comp_id, comp) for comp_id, comp in enumerate(team_comps)]

        # Filter teams consisting only of healer which may never win
        uids = self.get_uids(type=RoleTypes.HEALER, capability="role")
        build_plans = [plan for plan in build_plans if not all(map(lambda unit: unit["uid"] in uids, plan["units"]))]

        self.build_plans = build_plans
        self.teams = self._to_teams(build_plans)
        return self.teams

    def _compose_unique_units(self):
        """
        Build a pool of unique units defined by the provided characteristics.
        :return:
        """
        units = list(enumerate(list(itertools.product(*self.characteristics))))
        return map(lambda unit: {'uid': unit[0], 'role': unit[1][0], 'attack_type': unit[1][1]}, units)

    def sort_team_units(self, teams, uid=0):
        """

        :param teams: teams to sort units
        :param uid: UID to sort to top, shifting the sorting origin to this number/uid.
        Following UIDs will be sorted with distance to uid.
        :return:
        """
        for team in teams:
            team.units.sort(key=lambda x: math.fabs(x["uid"] - uid))
        return teams

    @staticmethod
    def _to_teams(plans: List[dict]) -> List[Team]:
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
    pass
    # n = 3
    # teams = TeamComposer(team_size=5, characteristics=[RoleTypes, UnitAttackTypes])
    # roles = teams[0].roles
    # print(teams[0])
    # print(roles)
    # uids = teams.get_uids(type=RoleTypes.HEALER, capability="role")
    # print(uids)
    # uid = teams.get_unique_uid(role_type=RoleTypes.HEALER, attack_type=UnitAttackTypes.RANGED)
    # print(uid)
    # ranged_healer_teams = teams.sample(5, contains=uid, unique=True)
    # for team in ranged_healer_teams:
    #     team.units.sort(key=lambda x: math.fabs(x["uid"] - uid))
    #
    # healer_teams = teams.sample(2, contains=uids)
    # print(healer_teams)
    # print(healer_teams[0].roles)
    # tids = healer_teams[0].get_team_ids(uids)
    # print(tids)
