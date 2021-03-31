from multiagent.core import RoleTypes, UnitAttackTypes
from multiprocess.connection import Connection, Pipe
from multiprocess.dummy import Manager, Process
from league.trainer_process import run
from league.utils.team_composer import TeamComposer


def _handle_match_results(parent_conn: Connection):
    msg = parent_conn.recv()
    if msg == "close":
        parent_conn.close()
    else:
        home, opponent, result = msg
        print(home, opponent, result)


def main():
    # Build league
    team_size = 3
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = team_composer.compose_unique_teams(team_size)[:2]  # TODO change back to all comps

    manager = Manager()
    payoff = manager.dict()

    # players_n = league.roles_per_initial_agent() * len(team_compositions) TODO: Uncomment if running on bigger hardware
    players_n = len(team_compositions)
    processes = []  # processes list - each representing a runner playing a match
    parent_conns = []  # parent connections

    # Start league
    for idx in range(players_n):
        parent_conn, child_conn = Pipe()
        parent_conns.append(parent_conn)

        proc = Process(target=run, args=(idx, payoff, child_conn))
        processes.append(proc)
        proc.start()

    # Receive match results
    while any(not parent_conn.closed for parent_conn in parent_conns):
        for parent_conn in parent_conns:
            if not parent_conn.closed:
                _handle_match_results(parent_conn)
                print(payoff)

    # Wait for processes to finish
    [proc.join() for proc in processes]
    exit(1)


if __name__ == '__main__':
    main()
