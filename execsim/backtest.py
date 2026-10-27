"""The lookahead-safe backtest engine.

Timeline for every bar ``i`` (this ordering is the whole ballgame):

1. **Match existing orders** that were submitted on an *earlier* bar against
   bar ``i``. Market orders execute at bar ``i``'s **open** (plus slippage);
   resting limit orders fill if bar ``i``'s range reaches their price.
2. **Mark to market** at bar ``i``'s **close**.
3. **Ask the strategy for decisions**, passing it history ``candles[: i+1]``
   only. Any orders it submits are queued and become eligible starting at bar
   ``i+1``.

Because a decision made at the close of bar ``i`` cannot be executed until bar
``i+1``, the simulator can never use a bar's own future information to trade
within it. There is no path by which ``on_bar`` can observe ``candles[i+1:]``.

The engine supports partial fills that carry across bars (a large market order
consumes a capped slice of each bar's volume until complete) and optional
order expiry for resting limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .execution.engine import ExecutionConfig, ExecutionModel
from .models import Candle, Fill, Order, OrderStatus, OrderType, Side
from .portfolio import EquityPoint, Portfolio, RealizedTrade
from .strategies.base import Strategy, StrategyContext
from .data.loader import infer_granularity


@dataclass
class BacktestResult:
    """Everything produced by a single strategy run over a dataset."""

    strategy_name: str
    starting_cash: float
    granularity: int
    candles: Sequence[Candle]
    equity_curve: List[EquityPoint]
    realized_trades: List[RealizedTrade]
    orders: List[Order]
    fills: List[Fill]
    portfolio: Portfolio
    exec_config: ExecutionConfig

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1].equity if self.equity_curve else self.starting_cash

    @property
    def total_fees(self) -> float:
        return self.portfolio.total_fees

    @property
    def num_fills(self) -> int:
        return len(self.fills)


class Backtester:
    def __init__(
        self,
        starting_cash: float = 100_000.0,
        exec_config: Optional[ExecutionConfig] = None,
        liquidate_at_end: bool = True,
    ) -> None:
        self.starting_cash = starting_cash
        self.exec_config = exec_config or ExecutionConfig()
        self.liquidate_at_end = liquidate_at_end

    def run(
        self,
        strategy: Strategy,
        candles: Sequence[Candle],
        granularity: Optional[int] = None,
    ) -> BacktestResult:
        if len(candles) < 2:
            raise ValueError("need at least 2 candles to run a backtest")
        granularity = granularity or infer_granularity(list(candles))

        model = ExecutionModel(self.exec_config)
        portfolio = Portfolio(self.starting_cash)
        active_orders: List[Order] = []
        all_orders: List[Order] = []
        all_fills: List[Fill] = []

        strategy.on_start(len(candles))

        for i, bar in enumerate(candles):
            # --- 1. Match previously-submitted orders against this bar -------
            still_active: List[Order] = []
            for order in active_orders:
                fill = self._match(order, bar, model)
                if fill is not None:
                    portfolio.apply_fill(fill)
                    all_fills.append(fill)

                order.bars_rested += 1
                if order.status is OrderStatus.FILLED:
                    continue
                if self._should_expire(order):
                    order.status = OrderStatus.EXPIRED
                    continue
                still_active.append(order)
            active_orders = still_active

            # --- 2. Mark to market at this bar's close -----------------------
            portfolio.mark(bar.ts, bar.close)

            # --- 3. Strategy decision using history up to and including bar i -
            ctx = StrategyContext(
                history=candles[: i + 1],
                index=i,
                position=portfolio.position,
                cash=portfolio.cash,
                equity=portfolio.equity(bar.close),
                open_orders=tuple(active_orders),
            )
            strategy.on_bar(ctx)

            if ctx._cancel_all:
                for order in active_orders:
                    if order.is_active:
                        order.status = OrderStatus.CANCELLED
                active_orders = []

            for order in ctx._submitted:
                order.status = OrderStatus.OPEN
                active_orders.append(order)
                all_orders.append(order)

        strategy.on_finish()

        # Optionally realize any open position at the final close so that
        # trade-based stats (win rate, profit factor) reflect the whole run.
        if self.liquidate_at_end and abs(portfolio.position) > 1e-12:
            last = candles[-1]
            closing = self._synthetic_liquidation(portfolio, last, model)
            if closing is not None:
                portfolio.apply_fill(closing)
                all_fills.append(closing)
                # Refresh the final equity point post-liquidation.
                portfolio.equity_curve[-1] = EquityPoint(
                    ts=last.ts,
                    price=last.close,
                    cash=portfolio.cash,
                    position=portfolio.position,
                    equity=portfolio.equity(last.close),
                )

        return BacktestResult(
            strategy_name=getattr(strategy, "name", type(strategy).__name__),
            starting_cash=self.starting_cash,
            granularity=granularity,
            candles=candles,
            equity_curve=portfolio.equity_curve,
            realized_trades=portfolio.realized_trades,
            orders=all_orders,
            fills=all_fills,
            portfolio=portfolio,
            exec_config=self.exec_config,
        )

    # -- internals ----------------------------------------------------------

    def _match(self, order: Order, bar: Candle, model: ExecutionModel) -> Optional[Fill]:
        if not order.is_active:
            return None
        if order.type is OrderType.MARKET:
            # Market orders decided last bar hit the market at this bar's open.
            return model.execute_market(order, bar, reference_price=bar.open, ts=bar.ts)
        return model.execute_limit(order, bar, ts=bar.ts)

    def _should_expire(self, order: Order) -> bool:
        if order.expire_after_bars is None:
            return False
        return order.bars_rested >= order.expire_after_bars

    def _synthetic_liquidation(
        self, portfolio: Portfolio, last: Candle, model: ExecutionModel
    ) -> Optional[Fill]:
        """Close any residual position at the final close (taker, with slippage)."""
        qty = abs(portfolio.position)
        if qty <= 1e-12:
            return None
        side = Side.SELL if portfolio.position > 0 else Side.BUY
        order = Order(side=side, qty=qty, type=OrderType.MARKET,
                      created_ts=last.ts, tag="final_liquidation")
        # Allow the whole position to close regardless of participation cap so
        # the run ends flat; slippage/fees still apply at the configured rates.
        prev_partial = model.config.allow_partial_fills
        prev_participation = model.config.max_participation
        model.config.allow_partial_fills = True
        model.config.max_participation = 1.0
        try:
            fill = model.execute_market(order, last, reference_price=last.close, ts=last.ts)
        finally:
            model.config.allow_partial_fills = prev_partial
            model.config.max_participation = prev_participation
        return fill


def run_backtest(
    strategy: Strategy,
    candles: Sequence[Candle],
    starting_cash: float = 100_000.0,
    exec_config: Optional[ExecutionConfig] = None,
    granularity: Optional[int] = None,
    liquidate_at_end: bool = True,
) -> BacktestResult:
    """Convenience wrapper around :class:`Backtester`."""
    return Backtester(
        starting_cash=starting_cash,
        exec_config=exec_config,
        liquidate_at_end=liquidate_at_end,
    ).run(strategy, candles, granularity=granularity)
