"""Portfolio accounting: cash, position, equity curve, and realized trades.

Uses average-cost accounting. Cash reflects every fill and fee immediately, so
``equity = cash + position * mark_price`` is always the true mark-to-market
value. Separately, we record *realized round-trip trades* each time a position
is reduced or closed, which drives win-rate and profit-factor metrics.

Fees are attributed to trades so that per-trade PnL is reported net of costs:
each closing event subtracts the exit fee plus the proportional entry fee that
was paid to establish the quantity being closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import Fill, Side


@dataclass(frozen=True)
class RealizedTrade:
    """A closed (or partially closed) position, net of fees."""

    entry_ts: int
    exit_ts: int
    side: Side            # direction of the position that was closed (BUY = was long)
    qty: float
    entry_price: float    # average cost basis, fee-exclusive
    exit_price: float
    gross_pnl: float      # (exit - entry) * qty * sign, before fees
    fees: float           # entry (prorated) + exit fees attributed to this close
    net_pnl: float        # gross_pnl - fees

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.qty
        return self.net_pnl / cost if cost else 0.0


@dataclass
class EquityPoint:
    ts: int
    price: float
    cash: float
    position: float
    equity: float


class Portfolio:
    def __init__(self, starting_cash: float = 100_000.0) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position = 0.0            # signed base-asset units
        self._avg_price = 0.0          # fee-exclusive cost basis per unit
        self._entry_fee_per_unit = 0.0
        self._entry_ts = 0
        self.total_fees = 0.0
        self.realized_trades: List[RealizedTrade] = []
        self.equity_curve: List[EquityPoint] = []

    # -- position helpers ---------------------------------------------------

    @property
    def avg_price(self) -> float:
        return self._avg_price

    def _pos_sign(self) -> int:
        if self.position > 0:
            return 1
        if self.position < 0:
            return -1
        return 0

    # -- fills --------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Update cash, position and realized trades from an execution."""
        signed_qty = fill.side.sign * fill.qty
        # Cash: buying spends quote, selling receives quote; fees always cost.
        self.cash -= fill.side.sign * fill.notional
        self.cash -= fill.fee
        self.total_fees += fill.fee

        prev_sign = self._pos_sign()
        incoming_sign = fill.side.sign

        if prev_sign == 0 or prev_sign == incoming_sign:
            # Opening or adding to a position in the same direction.
            self._add_to_position(fill)
        else:
            # Reducing / closing / possibly flipping the position.
            self._reduce_position(fill)

    def _add_to_position(self, fill: Fill) -> None:
        abs_pos = abs(self.position)
        new_abs = abs_pos + fill.qty
        if abs_pos == 0:
            self._entry_ts = fill.ts
        self._avg_price = (self._avg_price * abs_pos + fill.price * fill.qty) / new_abs
        self._entry_fee_per_unit = (
            self._entry_fee_per_unit * abs_pos + fill.fee
        ) / new_abs
        self.position += fill.side.sign * fill.qty

    def _reduce_position(self, fill: Fill) -> None:
        abs_pos = abs(self.position)
        pos_sign = self._pos_sign()
        close_qty = min(fill.qty, abs_pos)

        gross = (fill.price - self._avg_price) * close_qty * pos_sign
        exit_fee = fill.fee * (close_qty / fill.qty) if fill.qty else 0.0
        entry_fee = self._entry_fee_per_unit * close_qty
        net = gross - exit_fee - entry_fee

        self.realized_trades.append(
            RealizedTrade(
                entry_ts=self._entry_ts,
                exit_ts=fill.ts,
                side=Side.BUY if pos_sign > 0 else Side.SELL,
                qty=close_qty,
                entry_price=self._avg_price,
                exit_price=fill.price,
                gross_pnl=gross,
                fees=exit_fee + entry_fee,
                net_pnl=net,
            )
        )

        self.position += fill.side.sign * fill.qty
        remainder = fill.qty - close_qty

        if abs(self.position) <= 1e-12:
            # Fully flat.
            self.position = 0.0
            self._avg_price = 0.0
            self._entry_fee_per_unit = 0.0
        elif remainder > 1e-12 and self._pos_sign() == fill.side.sign:
            # Position flipped: the remainder opens a fresh position at fill price.
            self._avg_price = fill.price
            self._entry_fee_per_unit = (
                fill.fee * (remainder / fill.qty) / remainder if remainder else 0.0
            )
            self._entry_ts = fill.ts
        # else: still in the same direction, avg price/fee-per-unit unchanged.

    # -- marking ------------------------------------------------------------

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def mark(self, ts: int, price: float) -> EquityPoint:
        """Record a mark-to-market snapshot on the equity curve."""
        point = EquityPoint(
            ts=ts,
            price=price,
            cash=self.cash,
            position=self.position,
            equity=self.equity(price),
        )
        self.equity_curve.append(point)
        return point

    # -- summaries ----------------------------------------------------------

    def unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self._avg_price) * self.position

    def win_loss_counts(self) -> Tuple[int, int, int]:
        wins = sum(1 for t in self.realized_trades if t.net_pnl > 0)
        losses = sum(1 for t in self.realized_trades if t.net_pnl < 0)
        scratches = sum(1 for t in self.realized_trades if t.net_pnl == 0)
        return wins, losses, scratches
