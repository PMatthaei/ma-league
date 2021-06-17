from .simple_league import SimpleLeague
from .alpha_star_league import AlphaStarLeague

REGISTRY = {
    "simple": SimpleLeague,
    "alpha": AlphaStarLeague
}
