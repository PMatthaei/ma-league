from league.roles.simple.simple_league import SimpleLeague
from league.roles.alphastar.alpha_star_league import AlphaStarLeague

REGISTRY = {
    "simple": SimpleLeague,
    "alpha": AlphaStarLeague
}
