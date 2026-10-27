"""Mean-reversion strategy based on a rolling z-score of price."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Optional

from .base import Strategy, StrategyContext


class MeanReversion(Strategy):
    """Fade deviations from a rolling mean.

    Computes a z-score ``z = (price - mean) / std`` over ``lookback`` bars.
    Enters long when price is unusually cheap (``z <= -entry_z``) and exits once
    it reverts toward the mean (``z >= -exit_z``). Symmetrically shorts rich
    prices when ``allow_short`` is set.

    Parameters
    ----------
    lookback:
        Rolling window for the mean/standard deviation.
    entry_z, exit_z:
        Entry and exit z-score thresholds (positive numbers). ``exit_z`` is
        typically small (exit near the mean).
    allocation:
        Fraction of equity to deploy per position.
    allow_short:
        Whether to also fade rich prices with short positions.
    """

    def __init__(
        self,
        lookback: int = 30,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        allocation: float = 0.95,
        allow_short: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if entry_z <= 0 or exit_z < 0:
            raise ValueError("entry_z must be > 0 and exit_z >= 0")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allocation = allocation
        self.allow_short = allow_short
        self.name = name or f"mean_reversion(lb={lookback},z={entry_z})"

    def _zscore(self, closes) -> Optional[float]:
        window = closes[-self.lookback:]
        if len(window) < self.lookback:
            return None
        mean = fmean(window)
        std = pstdev(window)
        if std <= 1e-12:
            return None
        return (closes[-1] - mean) / std

    def on_bar(self, ctx: StrategyContext) -> None:
        closes = ctx.closes()
        z = self._zscore(closes)
        if z is None:
            return

        pos = ctx.position
        is_long = pos > 1e-12
        is_short = pos < -1e-12
        is_flat = not is_long and not is_short

        if is_flat:
            if z <= -self.entry_z:
                ctx.target_notional_fraction(self.allocation, tag="enter_long")
            elif self.allow_short and z >= self.entry_z:
                ctx.target_notional_fraction(-self.allocation, tag="enter_short")
        elif is_long:
            # Exit once price has reverted back up near/through the mean.
            if z >= -self.exit_z:
                ctx.close_position(tag="exit_long")
        elif is_short:
            if z <= self.exit_z:
                ctx.close_position(tag="exit_short")
