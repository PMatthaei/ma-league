from .training import EnsembleLeagueInstance, MultiAgentExperimentInstance, RoleBasedLeagueInstance, MatchmakingLeagueInstance


REGISTRY = {
    "ensemble": EnsembleLeagueInstance,
    "matchmaking": MatchmakingLeagueInstance,
    "rolebased": RoleBasedLeagueInstance
}