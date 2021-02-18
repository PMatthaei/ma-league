def remove_monotonic_suffix(win_rates, players):
    if not win_rates:
        return win_rates, players

    for i in range(len(win_rates) - 1, 0, -1):
        if win_rates[i - 1] < win_rates[i]:
            return win_rates[:i + 1], players[:i + 1]

    return np.array([]), []
