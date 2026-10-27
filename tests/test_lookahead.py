"""Regression tests for lookahead safety — the core correctness property.

If the engine ever leaked future information into a decision, these tests would
fail: a strategy's decision on bar ``i`` must depend only on candles ``0..i``,
so truncating the dataset after bar ``i`` must not change what the strategy did
on any bar ``<= i``.
"""

import unittest

from execsim.backtest import run_backtest
from execsim.data.loader import generate_synthetic
from execsim.models import OrderType, Side
from execsim.strategies.base import Strategy, StrategyContext


class RecordingStrategy(Strategy):
    """Records (index, history_len, price) and trades on a deterministic rule."""

    name = "recording"

    def __init__(self):
        self.seen = []

    def on_bar(self, ctx: StrategyContext):
        self.seen.append((ctx.index, len(ctx.history), ctx.price))
        # Deterministic, purely history-based decision.
        closes = ctx.closes()
        if len(closes) >= 3 and closes[-1] > closes[-3] and ctx.position == 0:
            ctx.buy_market(0.01, tag="buy")
        elif ctx.position > 0 and len(closes) >= 2 and closes[-1] < closes[-2]:
            ctx.close_position()


class TestLookaheadSafety(unittest.TestCase):
    def test_history_window_never_exceeds_current_index(self):
        candles = generate_synthetic(n=200, seed=1)
        strat = RecordingStrategy()
        run_backtest(strat, candles, liquidate_at_end=False)
        for index, hist_len, _ in strat.seen:
            # The strategy may only ever see candles[0 .. index].
            self.assertEqual(hist_len, index + 1)

    def test_truncating_future_does_not_change_past_decisions(self):
        candles = generate_synthetic(n=300, seed=2)
        cutoff = 180

        full = RecordingStrategy()
        run_backtest(full, candles, liquidate_at_end=False)

        truncated = RecordingStrategy()
        run_backtest(truncated, candles[:cutoff], liquidate_at_end=False)

        full_prefix = [row for row in full.seen if row[0] < cutoff]
        self.assertEqual(full_prefix, truncated.seen)

    def test_market_order_executes_at_next_bar_open_not_same_bar(self):
        """A decision at the close of bar i must fill no earlier than bar i+1."""

        class BuyOnBarOne(Strategy):
            name = "buy_on_bar_1"

            def on_bar(self, ctx: StrategyContext):
                if ctx.index == 1:
                    ctx.buy_market(1.0, tag="t")

        # Distinct opens per bar so we can identify the execution bar by price.
        from execsim.models import Candle
        candles = [
            Candle(ts=i * 60, open=100 + i, high=200 + i, low=50 + i,
                   close=100 + i, volume=1000.0)
            for i in range(5)
        ]
        res = run_backtest(BuyOnBarOne(), candles, liquidate_at_end=False)
        self.assertEqual(len(res.fills), 1)
        fill = res.fills[0]
        # Decision made at close of bar index 1 (ts=60) -> fills on bar index 2
        # (ts=120) at that bar's open (102), plus slippage.
        self.assertEqual(fill.ts, 120)
        self.assertGreaterEqual(fill.price, 102.0)
        self.assertLess(fill.price, 103.5)


if __name__ == "__main__":
    unittest.main()
