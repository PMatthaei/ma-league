from league.league import League
from league.roles.players import Player


class Coordinator:

    def __init__(self, league: League):
        """
        Central worker that maintains payoff matrix and assigns new matches.
        :param league:
        """
        self.league = league

    def send_outcome(self, home: Player, opponent: Player, outcome):
        """
        Update the payoff matrix
        :param home:
        :param opponent:
        :param outcome:
        :return:
        """
        self.league.update(home, opponent, outcome)
        if home.ready_to_checkpoint():
            self.league.add_player(opponent.checkpoint())
