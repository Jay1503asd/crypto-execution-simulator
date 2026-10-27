"""The strategy interface and the context object handed to strategies.

The ``StrategyContext`` is the *only* channel through which a strategy sees the
market and acts on it. By construction it exposes history **up to and including
the current bar and no further** — a strategy literally cannot read a future
candle, because future candles are not in the window the backtester passes in.

Strategies submit orders through the context. Those orders are queued and only
become eligible for execution on the *next* bar, which is enforced by the
backtest engine. This is the mechanism that prevents lookahead bias.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence

from ..models import Candle, Order, OrderType, Side


class StrategyContext:
    """Per-bar view of the world passed to :meth:`Strategy.on_bar`.

    Attributes
    ----------
    history:
        Read-only sequence of candles from the start of the dataset up to and
        including the current bar. Index ``-1`` is the current bar.
    index:
        Position of the current bar in the full dataset.
    position:
        Current signed base-asset position (``> 0`` long, ``< 0`` short).
    cash:
        Current quote-currency cash balance.
    """

    def __init__(
        self,
        history: Sequence[Candle],
        index: int,
        position: float,
        cash: float,
        equity: float,
        open_orders: Sequence[Order],
    ) -> None:
        self.history = history
        self.index = index
        self.position = position
        self.cash = cash
        self.equity = equity
        self.open_orders = open_orders
        self._submitted: List[Order] = []
        self._cancel_all = False

    # -- market data (current bar only) ------------------------------------

    @property
    def bar(self) -> Candle:
        return self.history[-1]

    @property
    def price(self) -> float:
        """Current reference price (close of the current bar)."""
        return self.history[-1].close

    @property
    def ts(self) -> int:
        return self.history[-1].ts

    def closes(self, lookback: Optional[int] = None) -> List[float]:
        """Closing prices, optionally just the most recent ``lookback`` bars."""
        series = [c.close for c in self.history]
        return series[-lookback:] if lookback else series

    # -- order submission ---------------------------------------------------

    def submit(self, order: Order) -> Order:
        self._submitted.append(order)
        return order

    def buy_market(self, qty: float, tag: str = "") -> Order:
        return self.submit(
            Order(side=Side.BUY, qty=qty, type=OrderType.MARKET, created_ts=self.ts, tag=tag)
        )

    def sell_market(self, qty: float, tag: str = "") -> Order:
        return self.submit(
            Order(side=Side.SELL, qty=qty, type=OrderType.MARKET, created_ts=self.ts, tag=tag)
        )

    def buy_limit(self, qty: float, price: float, expire_after_bars: Optional[int] = None,
                  tag: str = "") -> Order:
        return self.submit(
            Order(side=Side.BUY, qty=qty, type=OrderType.LIMIT, created_ts=self.ts,
                  limit_price=price, expire_after_bars=expire_after_bars, tag=tag)
        )

    def sell_limit(self, qty: float, price: float, expire_after_bars: Optional[int] = None,
                   tag: str = "") -> Order:
        return self.submit(
            Order(side=Side.SELL, qty=qty, type=OrderType.LIMIT, created_ts=self.ts,
                  limit_price=price, expire_after_bars=expire_after_bars, tag=tag)
        )

    def cancel_all_open(self) -> None:
        """Request cancellation of all currently resting orders before next bar."""
        self._cancel_all = True

    # -- sizing helpers -----------------------------------------------------

    def close_position(self, tag: str = "close") -> Optional[Order]:
        """Submit a market order to flatten the current position, if any."""
        if self.position > 1e-12:
            return self.sell_market(self.position, tag=tag)
        if self.position < -1e-12:
            return self.buy_market(-self.position, tag=tag)
        return None

    def target_position(self, target_qty: float, tag: str = "target") -> Optional[Order]:
        """Submit a market order to move the position toward ``target_qty``."""
        delta = target_qty - self.position
        if abs(delta) <= 1e-12:
            return None
        if delta > 0:
            return self.buy_market(delta, tag=tag)
        return self.sell_market(-delta, tag=tag)

    def target_notional_fraction(self, fraction: float, tag: str = "target") -> Optional[Order]:
        """Target a signed fraction of current equity in the asset.

        ``fraction=1.0`` means "go fully long with current equity"; ``0`` means
        flat; ``-0.5`` means short half of equity. Uses the current close as the
        sizing price (fills happen next bar, so actual exposure will differ
        slightly — realistic).
        """
        if self.price <= 0:
            return None
        target_qty = (fraction * self.equity) / self.price
        return self.target_position(target_qty, tag=tag)


class Strategy(abc.ABC):
    """Base class for all strategies.

    Subclasses implement :meth:`on_bar`. They may keep internal state across
    bars (that is realistic — it only ever reflects past data). They must never
    attempt to look beyond ``ctx.history``.
    """

    #: Human-readable name; override in subclasses or set per-instance.
    name: str = "strategy"

    def on_start(self, total_bars: int) -> None:
        """Called once before the first bar."""

    @abc.abstractmethod
    def on_bar(self, ctx: StrategyContext) -> None:
        """Called once per bar, in chronological order."""

    def on_finish(self) -> None:
        """Called once after the last bar."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
