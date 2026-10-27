"""The execution model: the difference between a backtest and reality.

This module turns an intended order into realistic fills against a single
historical bar, accounting for:

* **Fees** — maker vs. taker rates (market orders always pay taker; resting
  limit orders pay maker).
* **Slippage** — market orders cross the spread and move the price. We model a
  fixed half-spread cost plus a size-dependent *market-impact* term that scales
  with how much of the bar's volume the order consumes.
* **Partial fills** — an order cannot consume unlimited volume in one bar. A
  *participation cap* limits how much of a bar's traded volume a single order
  may take; the remainder carries to subsequent bars (or expires).

The model is deliberately *stateless with respect to time*: it attempts a fill
against whatever bar the backtester hands it. The backtester owns order
lifecycle (resting, expiry) and timing (which bar, and the reference price).
This separation is what keeps the simulation lookahead-safe and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import Candle, Fill, Liquidity, Order, OrderType, Side


@dataclass
class ExecutionConfig:
    """Tunable assumptions for the execution model.

    Fee defaults approximate Coinbase Advanced retail taker/maker pricing; set
    them to your actual tier. All rates are fractions (0.006 == 0.60%).
    """

    taker_fee: float = 0.006          # paid by market orders / crossing limits
    maker_fee: float = 0.002          # paid by resting limit orders
    half_spread: float = 0.0005       # baseline slippage for a taker (0.05%)
    impact_coefficient: float = 0.5   # market-impact slippage per unit participation
    max_participation: float = 0.10   # max fraction of a bar's volume one order may take
    allow_partial_fills: bool = True
    # Floor on bar volume (base units) to avoid div-by-zero / absurd impact on
    # thin/zero-volume bars. If a bar reports zero volume we treat liquidity as
    # this value.
    min_bar_volume: float = 1e-9

    def __post_init__(self) -> None:
        for name in ("taker_fee", "maker_fee", "half_spread", "impact_coefficient"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not (0 < self.max_participation <= 1.0):
            raise ValueError("max_participation must be in (0, 1]")


class ExecutionModel:
    """Produces realistic fills for market and limit orders against a bar."""

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.config = config or ExecutionConfig()

    # -- public API ---------------------------------------------------------

    def execute_market(
        self, order: Order, bar: Candle, reference_price: float, ts: int
    ) -> Optional[Fill]:
        """Attempt to fill a market order against ``bar``.

        ``reference_price`` is the price the backtester chose for this bar
        (typically the bar's open, since a decision made at the previous close
        hits the market at the next open). Slippage is applied on top.
        """
        if order.type is not OrderType.MARKET:
            raise ValueError("execute_market called with a non-market order")
        qty = self._fillable_qty(order.remaining_qty, bar)
        if qty <= 0:
            return None

        slip = self._slippage_fraction(qty, reference_price, bar)
        # Buyers pay up, sellers receive less.
        fill_price = reference_price * (1.0 + order.side.sign * slip)
        fill_price = max(fill_price, 1e-12)
        fee = fill_price * qty * self.config.taker_fee
        fill = Fill(
            order_id=order.id,
            ts=ts,
            side=order.side,
            qty=qty,
            price=fill_price,
            fee=fee,
            liquidity=Liquidity.TAKER,
        )
        order.record_fill(fill)
        return fill

    def execute_limit(self, order: Order, bar: Candle, ts: int) -> Optional[Fill]:
        """Attempt to fill a resting limit order against ``bar``.

        A buy limit fills only if the bar trades down to the limit
        (``bar.low <= limit``); a sell limit fills only if the bar trades up to
        it (``bar.high >= limit``). If the bar *gaps through* the limit at the
        open, the order fills at the (better) open price — realistic price
        improvement for a resting order. Resting limits earn the maker fee and
        incur no slippage: they are executed at their own price.
        """
        if order.type is not OrderType.LIMIT:
            raise ValueError("execute_limit called with a non-limit order")
        assert order.limit_price is not None
        limit = order.limit_price

        if order.side is Side.BUY:
            if bar.low > limit:
                return None  # price never reached the bid
            # Gap down through the limit -> fill at the open (price improvement).
            fill_price = min(limit, bar.open)
        else:  # SELL
            if bar.high < limit:
                return None  # price never reached the ask
            fill_price = max(limit, bar.open)

        qty = self._fillable_qty(order.remaining_qty, bar)
        if qty <= 0:
            return None

        fee = fill_price * qty * self.config.maker_fee
        fill = Fill(
            order_id=order.id,
            ts=ts,
            side=order.side,
            qty=qty,
            price=fill_price,
            fee=fee,
            liquidity=Liquidity.MAKER,
        )
        order.record_fill(fill)
        return fill

    # -- helpers (exposed for testing/transparency) -------------------------

    def _bar_liquidity(self, bar: Candle) -> float:
        return max(bar.volume, self.config.min_bar_volume)

    def _fillable_qty(self, requested: float, bar: Candle) -> float:
        """Cap the requested quantity at the bar's participation limit."""
        if requested <= 0:
            return 0.0
        cap = self.config.max_participation * self._bar_liquidity(bar)
        if not self.config.allow_partial_fills:
            # All-or-nothing within the cap: if we cannot fill it all, fill none.
            return requested if requested <= cap else 0.0
        return min(requested, cap)

    def _slippage_fraction(self, qty: float, reference_price: float, bar: Candle) -> float:
        """Slippage as a fraction of reference price for a taker order.

        ``half_spread + impact_coefficient * participation`` where
        ``participation = order_qty / bar_volume``. Larger orders relative to
        the bar's liquidity suffer more impact.
        """
        participation = qty / self._bar_liquidity(bar)
        return self.config.half_spread + self.config.impact_coefficient * participation
