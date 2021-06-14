import numpy as np


def _get_stat(episodal_stats, k, current_stats):
    # Check if the key exists in on of the dicts and retrieve the type of the corresponding value
    if k in current_stats:
        stat_type = type(current_stats[k])
    elif k in episodal_stats:
        stat_type = type(episodal_stats[k])
    else:
        raise KeyError(f"Key {k} not found in supplied env_info dict which is used to update the current stats.")

    if stat_type is int or stat_type is float or stat_type is bool:
        return current_stats.get(k, 0)
    elif stat_type is list:
        return np.array(current_stats.get(k, []), dtype=int)
    elif isinstance(current_stats[k], (np.ndarray, np.generic)):
        return current_stats.get(k, np.array([]))


def _update_stats(episodal_stats, k, current_stats):
    """
    Integrate environment information into stats dict depending on the incoming data type.
    :param k:
    :param current_stats:
    :return:
    """
    # Check if the key exists in on of the dicts and retrieve the type of the corresponding value
    if k in current_stats:
        stat_type = type(current_stats[k])
    elif k in episodal_stats:
        stat_type = type(episodal_stats[k])
    else:
        raise KeyError("Key not found in supplied env_info dict which is used to update the current stats.")

    # Define how data should be merged depending on the type into a single dict and which default value to use if data
    # does not yet exists in one of the merging dicts
    if stat_type is int or stat_type is float or stat_type is bool:
        return episodal_stats.get(k, 0) + current_stats.get(k, 0)
    elif stat_type is list:
        return np.array(episodal_stats.get(k, np.zeros_like(current_stats.get(k))), dtype=int) + np.array(
            current_stats.get(k, []), dtype=int)


def precollect_stats(parallel, episodal_stats, current_stats):
    stats_dict = {}
    if parallel:
        if len(episodal_stats) == 0 and len(current_stats) > 0:
            infos = current_stats
        elif len(episodal_stats) > 0 and len(current_stats) == 0:
            infos = [episodal_stats]
        else:
            infos = [episodal_stats] + current_stats

        stats_dict.update({k: np.sum(_get_stat(episodal_stats, k, d) for d in infos) for k in set.union(*[set(d) for d in infos])})
        #stats_dict["n_episodes"] = batch_size + episodal_stats.get("n_episodes", 0)
        #stats_dict["ep_length"] = sum(ep_lens) + episodal_stats.get("ep_length", 0)
    else:
        stats_dict.update({k: _update_stats(episodal_stats, k, current_stats) for k in set(episodal_stats) | set(current_stats)})
        #stats_dict["n_episodes"] = 1 + episodal_stats.get("n_episodes", 0)
        #stats_dict["ep_length"] = t + episodal_stats.get("ep_length", 0)
    return stats_dict
