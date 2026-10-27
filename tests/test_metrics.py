import unittest

from execsim.backtest import run_backtest
from execsim.data.loader import generate_synthetic
from execsim.execution.engine import ExecutionConfig
from execsim.metrics import SECONDS_PER_YEAR, compute_metrics, _max_drawdown
from execsim.models import Candle
from execsim.portfolio import EquityPoint
from execsim.strategies.buy_and_hold import BuyAndHold


def eqpt(ts, equity):
    return EquityPoint(ts=ts, price=equity, cash=0.0, position=0.0, equity=equity)


class TestDrawdown(unittest.TestCase):
    def test_simple_drawdown(self):
        curve = [eqpt(0, 100), eqpt(1, 120), eqpt(2, 60), eqpt(3, 90)]
        dd = _max_drawdown(curve)
        # Peak 120 -> trough 60 == -50%.
        self.assertAlmostEqual(dd.max_drawdown, 0.5)

    def test_recovery_detected(self):
        curve = [eqpt(0, 100), eqpt(1, 80), eqpt(2, 100), eqpt(3, 130)]
        dd = _max_drawdown(curve)
        self.assertAlmostEqual(dd.max_drawdown, 0.2)
        self.assertEqual(dd.recovery_ts, 2)


class TestMetrics(unittest.TestCase):
    def test_buy_and_hold_return_tracks_price(self):
        # Flat-cost world so return ~ price change on an all-in buy-and-hold.
        candles = [
            Candle(ts=i * 3600, open=100.0, high=100.0, low=100.0,
                   close=100.0 + i, volume=1e9)
            for i in range(50)
        ]
        cfg = ExecutionConfig(taker_fee=0.0, maker_fee=0.0, half_spread=0.0,
                              impact_coefficient=0.0, max_participation=1.0)
        res = run_backtest(BuyAndHold(allocation=1.0), candles, starting_cash=10_000.0,
                           exec_config=cfg, liquidate_at_end=False)
        m = compute_metrics(res)
        # Entry at bar 1 open (100), final close 149 => ~49% on the invested amount.
        self.assertGreater(m.total_return, 0.40)
        self.assertLess(m.total_return, 0.50)
        self.assertEqual(m.bars, 50)

    def test_periods_per_year_matches_granularity(self):
        candles = generate_synthetic(n=100, granularity=3600, seed=3)
        res = run_backtest(BuyAndHold(), candles)
        m = compute_metrics(res)
        self.assertAlmostEqual(m.periods_per_year, SECONDS_PER_YEAR / 3600)

    def test_fees_make_buy_and_hold_underperform_price(self):
        candles = generate_synthetic(n=200, seed=4)
        free = ExecutionConfig(taker_fee=0.0, half_spread=0.0, impact_coefficient=0.0,
                               max_participation=1.0)
        costly = ExecutionConfig(taker_fee=0.02, half_spread=0.01, impact_coefficient=1.0,
                                 max_participation=1.0)
        r_free = run_backtest(BuyAndHold(), candles, exec_config=free, liquidate_at_end=False)
        r_cost = run_backtest(BuyAndHold(), candles, exec_config=costly, liquidate_at_end=False)
        self.assertGreater(compute_metrics(r_free).total_return,
                           compute_metrics(r_cost).total_return)


if __name__ == "__main__":
    unittest.main()
