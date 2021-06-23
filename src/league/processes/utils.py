from league.components.payoff import PayoffEntry


def extract_result(env_info: dict, policy_team_id: int):
    draw = env_info["draw"]
    battle_won = env_info["battle_won"]
    if draw or all(battle_won) or not any(battle_won):
        # Draw if all won or all lost
        result = PayoffEntry.DRAW
    elif battle_won[policy_team_id]:
        result = PayoffEntry.WIN
    else:
        result = PayoffEntry.LOSS
    return result