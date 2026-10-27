"""Buy-and-hold: the benchmark every active strategy must beat after costs."""

from __future__ import annotations

from .base import Strategy, StrategyContext


class BuyAndHold(Strategy):
    """Buy once on the first bar with (almost) all cash, then hold.

    A small cash buffer is left unspent so the single taker fee and slippage on
    the entry do not overdraw the account. This is the honest benchmark: it
    pays one round of costs, nothing more.
    """

    def __init__(self, allocation: float = 0.99, name: str = "buy_and_hold") -> None:
        if not (0 < allocation <= 1.0):
            raise ValueError("allocation must be in (0, 1]")
        self.allocation = allocation
        self.name = name
        self._entered = False

    def on_bar(self, ctx: StrategyContext) -> None:
        if self._entered:
            return
        if ctx.price <= 0:
            return
        qty = (self.allocation * ctx.cash) / ctx.price
        if qty > 0:
            ctx.buy_market(qty, tag="entry")
            self._entered = True
