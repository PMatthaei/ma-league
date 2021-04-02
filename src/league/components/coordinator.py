from league.league import League


class Coordinator:

    def __init__(self, league: League):
        """
        Central worker that maintains payoff matrix and assigns new matches.
        :param league:
        """
        self.league = league

    def send_outcome(self, home: int, away: int, outcome):
        """
        Update the payoff matrix
        :param home:
        :param away:
        :param outcome:
        :return:
        """
        home_player, _ = self.league.update(home, away, outcome)
        if home_player.ready_to_checkpoint():
            self.league.add_player(home_player.checkpoint())
