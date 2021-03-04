import itertools

from multiagent.core import RoleTypes, UnitAttackTypes

from utils.logging import get_logger


class TeamComposer:
    def __init__(self, attack_types=UnitAttackTypes, role_types=RoleTypes):
        self.logger = get_logger()
        self.attack_types = attack_types
        self.role_types = role_types

    def compose_unique_teams(self, team_size=1):
        units = self.compose_unique_units()
        team_comps = list(itertools.combinations_with_replacement(units, team_size))
        self.logger.info("Created {} unique team compositions with team size {}.".format(len(team_comps), team_size))
        return team_comps

    def compose_unique_units(self):
        build_data = [self.role_types, self.attack_types]
        units = list(itertools.product(*build_data))
        self.logger.info("Created {} unique units.".format(len(units)))
        return units


def main():
    n = 25
    TeamComposer().compose_unique_teams(n)


if __name__ == '__main__':
    main()
