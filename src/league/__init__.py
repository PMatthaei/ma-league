from league.rolebased.simple.simple_league import SimpleLeague
from league.rolebased.alphastar.alpha_star_league import AlphaStarLeague

REGISTRY = {
    "simple": SimpleLeague,
    "alpha": AlphaStarLeague
}
