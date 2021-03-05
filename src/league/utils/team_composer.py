import itertools

import enum

from multiagent.core import RoleTypes, UnitAttackTypes

from utils.logging import get_logger


class TeamComposer:
    def __init__(self, *characteristics: [enum.EnumMeta]):
        self.logger = get_logger()
        assert characteristics is not None and len(
            characteristics) > 0, "Please supply characteristics to create units from."
        self.characteristics = characteristics

    def compose_unique_teams(self, team_size=1):
        units = self.compose_unique_units()
        team_comps = list(itertools.combinations_with_replacement(units, team_size))
        self.logger.debug("Created {} unique team compositions with team size {}.".format(len(team_comps), team_size))
        return list(map(self._to_team_build_plan, team_comps))

    def compose_unique_units(self):
        self.logger.debug("Composing units with {} characteristics:".format(len(self.characteristics)))
        self.logger.debug((["Name {}: - Size: {}".format(c, len(c)) for c in self.characteristics]))
        units = list(itertools.product(*self.characteristics))
        self.logger.debug("Created {} unique units.".format(len(units)))
        return units

    def _to_team_build_plan(self, units, is_scripted=False):
        return {  # Team
            "is_scripted": is_scripted,
            "units": units
        }


if __name__ == '__main__':
    n = 3
    teams = TeamComposer(RoleTypes, UnitAttackTypes).compose_unique_teams(n)
    team_hash = hash(str(teams[0]))
    print(teams[0])
    print("Creating build plan for team with hash {}".format(team_hash))