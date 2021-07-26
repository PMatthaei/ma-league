from .training.ensemble_instance import EnsembleLeagueProcess
from .training.matchmaking_league_instance import MatchmakingLeagueInstance
from .training.role_based_league_instance import RolebasedLeagueProcess

REGISTRY = {
    "ensemble": EnsembleLeagueProcess,
    "matchmaking": MatchmakingLeagueInstance,
    "rolebased": RolebasedLeagueProcess
}