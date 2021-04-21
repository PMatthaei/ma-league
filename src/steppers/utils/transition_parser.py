def append_pre_transition_data(away_pre_transition_data, home_pre_transition_data, data):
    state = data["state"]
    actions = data["avail_actions"]
    obs = data["obs"]
    # TODO: only supports same team sizes!
    home_avail_actions = actions[:len(actions) // 2]
    home_obs = obs[:len(obs) // 2]
    home_pre_transition_data["state"].append(state)
    home_pre_transition_data["avail_actions"].append(home_avail_actions)
    home_pre_transition_data["obs"].append(home_obs)

    away_avail_actions = actions[len(actions) // 2:]
    away_obs = obs[len(obs) // 2:]
    away_pre_transition_data["state"].append(state)
    away_pre_transition_data["avail_actions"].append(away_avail_actions)
    away_pre_transition_data["obs"].append(away_obs)
