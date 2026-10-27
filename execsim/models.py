"""Core domain models shared across the simulator.

These are intentionally small, explicit value objects. Candles are immutable;
orders are mutable because the execution engine updates their fill state as the
simulation progresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import itertools


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for a buy (increases position), -1 for a sell."""
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"          # accepted, not yet eligible for matching
    OPEN = "open"                # resting / eligible for matching
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED)


class Liquidity(str, Enum):
    """Whether a fill took liquidity (taker) or provided it (maker)."""

    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar.

    ``ts`` is the bar's open time as a Unix timestamp (seconds, UTC).
    """

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float  # base-asset volume traded during the bar

    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True)
class Fill:
    """An execution against an order."""

    order_id: int
    ts: int
    side: Side
    qty: float                 # base-asset quantity filled (always positive)
    price: float               # execution price (quote per base), slippage included
    fee: float                 # quote-currency fee paid
    liquidity: Liquidity

    @property
    def notional(self) -> float:
        """Signed quote value of the fill, excluding fees."""
        return self.qty * self.price


_order_ids = itertools.count(1)


@dataclass
class Order:
    """A trading order.

    Quantities are in base asset units. ``limit_price`` is required for LIMIT
    orders and ignored for MARKET orders. ``created_ts`` is the bar close time
    at which the strategy decided to place the order; the engine guarantees the
    order can only be matched on *later* bars.
    """

    side: Side
    qty: float
    type: OrderType
    created_ts: int
    limit_price: Optional[float] = None
    # Number of bars the order may rest before expiring (None = good-till-cancel).
    expire_after_bars: Optional[int] = None
    tag: str = ""

    # --- runtime state (managed by the execution engine) ---
    id: int = field(default_factory=lambda: next(_order_ids))
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0
    bars_rested: int = 0
    fills: List[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"order qty must be positive, got {self.qty}")
        if self.type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require a limit_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive")

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return not self.status.is_terminal

    def record_fill(self, fill: Fill) -> None:
        """Apply a fill, updating average price, fees and status."""
        if fill.qty <= 0:
            return
        prev_notional = self.avg_fill_price * self.filled_qty
        self.filled_qty += fill.qty
        self.avg_fill_price = (prev_notional + fill.price * fill.qty) / self.filled_qty
        self.fees_paid += fill.fee
        self.fills.append(fill)
        # Use a tiny epsilon to absorb floating-point dust.
        if self.remaining_qty <= 1e-12:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
