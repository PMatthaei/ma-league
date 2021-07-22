from league.components.payoff_role_based import PayoffEntry


def extract_result(env_info: dict, policy_team_id: int):
    draw = env_info["draw"]
    battle_won = env_info["battle_won"]
    if draw or all(battle_won) or not any(battle_won):
        result = PayoffEntry.DRAW  # Draw if all won or all lost
    elif battle_won[policy_team_id]:  # Policy team(= home team) won
        result = PayoffEntry.WIN
    else:
        result = PayoffEntry.LOSS  # Policy team(= home team) lost
    return result
