from .training.ensemble_instance import EnsembleLeagueInstance
from .training.matchmaking_league_instance import MatchmakingLeagueInstance
from .training.role_based_league_instance import RolebasedLeagueProcess

REGISTRY = {
    "ensemble": EnsembleLeagueInstance,
    "matchmaking": MatchmakingLeagueInstance,
    "rolebased": RolebasedLeagueProcess
}