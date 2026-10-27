"""Moving-average crossover trend-following strategy."""

from __future__ import annotations

from statistics import fmean
from typing import Optional

from .base import Strategy, StrategyContext


def _sma(values, window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return fmean(values[-window:])


class MovingAverageCrossover(Strategy):
    """Go long when the fast SMA is above the slow SMA; flat (or short) below.

    Trades only on *state changes* (the actual crossover), not every bar, which
    keeps turnover — and therefore fees — realistic. Sizing targets a fraction
    of current equity so position size tracks account growth.

    Parameters
    ----------
    fast, slow:
        SMA windows in bars (``fast < slow``).
    allocation:
        Fraction of equity to deploy when in a position (0..1).
    allow_short:
        If True, take a short position on a bearish crossover instead of going
        flat.
    """

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        allocation: float = 0.95,
        allow_short: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("windows must be >= 1")
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        if not (0 < allocation <= 1.0):
            raise ValueError("allocation must be in (0, 1]")
        self.fast = fast
        self.slow = slow
        self.allocation = allocation
        self.allow_short = allow_short
        self.name = name or f"ma_crossover({fast}/{slow})"
        self._desired_fraction: float = 0.0

    def on_bar(self, ctx: StrategyContext) -> None:
        closes = ctx.closes()
        if len(closes) < self.slow:
            return  # not enough history to form the slow average yet

        fast_ma = _sma(closes, self.fast)
        slow_ma = _sma(closes, self.slow)
        if fast_ma is None or slow_ma is None:
            return

        if fast_ma > slow_ma:
            desired = self.allocation
        elif fast_ma < slow_ma:
            desired = -self.allocation if self.allow_short else 0.0
        else:
            desired = self._desired_fraction  # unchanged on an exact tie

        if desired != self._desired_fraction:
            self._desired_fraction = desired
            if desired == 0.0:
                ctx.close_position(tag="exit")
            else:
                ctx.target_notional_fraction(desired, tag="signal")
