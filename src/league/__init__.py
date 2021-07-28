from league.roles.alphastar.alpha_star_league import AlphaStarLeague
from league.roles.simple.simple_league import SimpleLeague

REGISTRY = {
    "simple": SimpleLeague,
    "alpha": AlphaStarLeague
}
