"""Trading strategies and the lookahead-safe strategy interface."""

from .base import Strategy, StrategyContext
from .buy_and_hold import BuyAndHold
from .ma_crossover import MovingAverageCrossover
from .mean_reversion import MeanReversion

# Registry used by the CLI to construct strategies by name.
REGISTRY = {
    "buyhold": BuyAndHold,
    "ma_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversion,
}

__all__ = [
    "Strategy",
    "StrategyContext",
    "BuyAndHold",
    "MovingAverageCrossover",
    "MeanReversion",
    "REGISTRY",
]
