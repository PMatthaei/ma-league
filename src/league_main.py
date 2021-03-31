import time

from multiagent.core import RoleTypes, UnitAttackTypes
from multiprocessing.connection import Connection, Pipe
from multiprocessing.dummy import Manager, Process

from league.league import League
from league.payoff import Payoff
from league.trainer_process import run
from league.utils.team_composer import TeamComposer


def _handle_messages(parent_conn: Connection, payoff: Payoff):
    msg = parent_conn.recv()
    if "close" in msg:
        print(f"Closing connection to process {msg['close']}")
        parent_conn.close()
    elif "result" in msg:
        home, away, result = msg["result"]
        payoff.update(home, away, result)
    else:
        raise Exception("Unknown message.")


def main():
    # Build league teams
    team_size = 3
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = team_composer.compose_unique_teams(team_size)[:2]  # TODO change back to all comps

    # Shared objects
    manager = Manager()
    p_matrix = manager.dict()
    players = manager.list()

    processes = []  # processes list - each representing a runner playing a match
    league_conns = []  # parent connections

    # Create league
    payoff = Payoff(p_matrix=p_matrix, players=players)
    league = League(initial_agents=team_compositions, payoff=payoff)

    players_n = league.roles_per_initial_agent() * len(team_compositions)

    # Start league training
    for idx in range(players_n):
        league_conn, conn = Pipe()
        league_conns.append(league_conn)

        player = league.get_player(idx)

        proc = Process(target=run, args=(idx, player, conn))
        processes.append(proc)
        proc.start()

    # Receive messages from all processes
    while any(not parent_conn.closed for parent_conn in league_conns):
        [_handle_messages(league_conn, payoff) for league_conn in league_conns if
         not league_conn.closed and league_conn.poll()]

    print(payoff.p_matrix)

    # Wait for processes to finish
    [proc.join() for proc in processes]
    exit(1)


if __name__ == '__main__':
    main()
