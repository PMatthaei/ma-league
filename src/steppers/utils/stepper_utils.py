from typing import List


def build_pre_transition_data(env):
    state = env.get_state()
    avail_actions = env.get_avail_actions()
    obs = env.get_obs()
    n_avail_actions = len(avail_actions)
    assert n_avail_actions % 2 == 0, f"{n_avail_actions} actions do not fit in the symmetric two-team scenario."

    home_avail_actions = avail_actions[:n_avail_actions // 2]
    home_obs = obs[:len(obs) // 2]
    home_pre_transition_data = {
        "state": [state],
        "avail_actions": [home_avail_actions],
        "obs": [home_obs]
    }
    opponent_avail_actions = avail_actions[n_avail_actions // 2:]
    opponent_obs = obs[len(obs) // 2:]
    opponent_pre_transition_data = {
        "state": [state],
        "avail_actions": [opponent_avail_actions],
        "obs": [opponent_obs]
    }
    return home_pre_transition_data, opponent_pre_transition_data


def append_pre_transition_data(away_pre_transition_data, home_pre_transition_data, data):
    state = data["state"]
    avail_actions = data["avail_actions"]
    obs = data["obs"]
    n_avail_actions = len(avail_actions)
    assert n_avail_actions % 2 == 0, f"{n_avail_actions} avail_actions do not fit in the symmetric two-team scenario."

    home_avail_actions = avail_actions[:n_avail_actions // 2]
    home_obs = obs[:len(obs) // 2]
    home_pre_transition_data["state"].append(state)
    home_pre_transition_data["avail_actions"].append(home_avail_actions)
    home_pre_transition_data["obs"].append(home_obs)

    away_avail_actions = avail_actions[n_avail_actions // 2:]
    away_obs = obs[len(obs) // 2:]
    away_pre_transition_data["state"].append(state)
    away_pre_transition_data["avail_actions"].append(away_avail_actions)
    away_pre_transition_data["obs"].append(away_obs)


def get_policy_team_id(teams: List):
    """
    Returns the ID of the first policy team found within a match build plan.
    This assumes not more than one training (=> is_scripted=False) team is playing.
    :param teams:
    :return:
    """
    return teams.index(next(filter(lambda x: not x["is_scripted"], teams), None))
