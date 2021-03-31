import time

from multiagent.core import RoleTypes, UnitAttackTypes
from multiprocessing.connection import Connection, Pipe
from multiprocessing.dummy import Manager, Process

from league.league import League
from league.payoff import Payoff
from league.trainer_process import run
from league.utils.team_composer import TeamComposer


def _handle_messages(parent_conn: Connection, payoff: Payoff):
    if parent_conn.poll():
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

    # players_n = league.roles_per_initial_agent() * len(team_compositions)
    players_n = len(team_compositions)
    processes = []  # processes list - each representing a runner playing a match
    parent_conns = []  # parent connections

    # Create league
    payoff = Payoff(p_matrix=p_matrix, players=players)
    league = League(initial_agents=team_compositions, payoff=payoff)

    # Start league training
    for idx in range(players_n):
        parent_conn, child_conn = Pipe()
        parent_conns.append(parent_conn)

        player = league.get_player(idx)

        proc = Process(target=run, args=(idx, player, child_conn))
        processes.append(proc)
        proc.start()

    # Receive messages from all processes
    while any(not parent_conn.closed for parent_conn in parent_conns):
        [_handle_messages(parent_conn, payoff) for parent_conn in parent_conns if not parent_conn.closed]

    print(payoff.p_matrix)

    # Wait for processes to finish
    [proc.join() for proc in processes]
    exit(1)


if __name__ == '__main__':
    main()
