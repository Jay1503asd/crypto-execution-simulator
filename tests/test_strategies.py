import unittest

from execsim.backtest import run_backtest
from execsim.data.loader import generate_synthetic
from execsim.execution.engine import ExecutionConfig
from execsim.models import Candle
from execsim.strategies.buy_and_hold import BuyAndHold
from execsim.strategies.ma_crossover import MovingAverageCrossover
from execsim.strategies.mean_reversion import MeanReversion


FREE = ExecutionConfig(taker_fee=0.0, maker_fee=0.0, half_spread=0.0,
                       impact_coefficient=0.0, max_participation=1.0)


class TestBuyAndHold(unittest.TestCase):
    def test_enters_once_and_holds(self):
        candles = generate_synthetic(n=100, seed=5)
        res = run_backtest(BuyAndHold(), candles, exec_config=FREE, liquidate_at_end=False)
        entries = [o for o in res.orders if o.tag == "entry"]
        self.assertEqual(len(entries), 1)
        self.assertGreater(res.portfolio.position, 0.0)


class TestMovingAverageCrossover(unittest.TestCase):
    def test_goes_long_on_uptrend(self):
        # Steadily rising price => fast MA above slow MA => long.
        candles = [
            Candle(ts=i * 3600, open=100 + i, high=101 + i, low=99 + i,
                   close=100 + i, volume=1e9)
            for i in range(120)
        ]
        res = run_backtest(MovingAverageCrossover(fast=5, slow=20), candles,
                           exec_config=FREE, liquidate_at_end=False)
        self.assertGreater(res.portfolio.position, 0.0)

    def test_validates_params(self):
        with self.assertRaises(ValueError):
            MovingAverageCrossover(fast=50, slow=20)


class TestMeanReversion(unittest.TestCase):
    def test_buys_the_dip(self):
        # Flat price then a sharp dip on the last bar => negative z-score => buy.
        closes = [100.0] * 40 + [80.0]
        candles = [
            Candle(ts=i * 3600, open=c, high=c + 0.5, low=c - 0.5, close=c, volume=1e9)
            for i, c in enumerate(closes)
        ]
        res = run_backtest(MeanReversion(lookback=20, entry_z=1.0, exit_z=0.2),
                           candles, exec_config=FREE, liquidate_at_end=False)
        buys = [o for o in res.orders if o.tag == "enter_long"]
        self.assertGreaterEqual(len(buys), 1)

    def test_validates_params(self):
        with self.assertRaises(ValueError):
            MeanReversion(lookback=1)


class TestMultiStrategySameData(unittest.TestCase):
    def test_runs_independently_on_same_candles(self):
        candles = generate_synthetic(n=300, seed=6)
        strategies = [
            BuyAndHold(),
            MovingAverageCrossover(fast=10, slow=30),
            MeanReversion(lookback=20),
        ]
        results = [run_backtest(s, candles) for s in strategies]
        self.assertEqual(len(results), 3)
        # Each run is isolated: same candle objects, independent portfolios.
        ids = {id(r.portfolio) for r in results}
        self.assertEqual(len(ids), 3)
        for r in results:
            self.assertEqual(len(r.equity_curve), 300)


if __name__ == "__main__":
    unittest.main()
