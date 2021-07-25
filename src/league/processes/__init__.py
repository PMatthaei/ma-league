from .training.ensemble_league_process import EnsembleLeagueProcess
from .training.matchmaking_league_process import MatchmakingLeagueProcess
from .training.role_based_league_process import RolebasedLeagueProcess

REGISTRY = {
    "ensemble": EnsembleLeagueProcess,
    "matchmaking": MatchmakingLeagueProcess,
    "rolebased": RolebasedLeagueProcess
}