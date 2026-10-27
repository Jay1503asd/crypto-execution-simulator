"""execsim: a realistic crypto execution simulator and strategy backtester.

A zero-dependency (standard-library-only) toolkit for simulating how trading
strategies would have performed against historical Coinbase market data, with
an emphasis on *realistic execution* rather than idealized fills.

Key properties
--------------
- Lookahead-safe backtest engine: a strategy only ever sees candles up to and
  including the bar on which it is making a decision. Orders decided at the
  close of bar ``t`` are executed starting at bar ``t+1``.
- Explicit execution modeling: transaction fees (maker/taker), slippage with
  market impact, and volume-constrained partial fills.
- Risk-aware metrics: drawdown, Sharpe, Sortino, win rate, profit factor, and a
  built-in comparison against buy-and-hold.
"""

from .models import Candle, Order, Fill, Side, OrderType, OrderStatus, Liquidity

__all__ = [
    "Candle",
    "Order",
    "Fill",
    "Side",
    "OrderType",
    "OrderStatus",
    "Liquidity",
]

__version__ = "0.1.0"
