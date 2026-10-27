import unittest

from execsim.execution.engine import ExecutionConfig, ExecutionModel
from execsim.models import Candle, Order, OrderType, Side, Liquidity


def bar(o=100.0, h=110.0, l=90.0, c=105.0, v=100.0, ts=0):
    return Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v)


class TestMarketExecution(unittest.TestCase):
    def test_buy_pays_up_sell_receives_less(self):
        model = ExecutionModel(ExecutionConfig(half_spread=0.001, impact_coefficient=0.0,
                                               taker_fee=0.0, max_participation=1.0))
        b = bar(o=100.0, v=1000.0)
        buy = Order(Side.BUY, 1.0, OrderType.MARKET, created_ts=0)
        sell = Order(Side.SELL, 1.0, OrderType.MARKET, created_ts=0)
        fb = model.execute_market(buy, b, reference_price=100.0, ts=0)
        fs = model.execute_market(sell, b, reference_price=100.0, ts=0)
        self.assertGreater(fb.price, 100.0)       # buyer pays more than reference
        self.assertLess(fs.price, 100.0)          # seller gets less
        self.assertAlmostEqual(fb.price, 100.0 * 1.001)
        self.assertAlmostEqual(fs.price, 100.0 * 0.999)
        self.assertEqual(fb.liquidity, Liquidity.TAKER)

    def test_impact_scales_with_participation(self):
        model = ExecutionModel(ExecutionConfig(half_spread=0.0, impact_coefficient=1.0,
                                               taker_fee=0.0, max_participation=1.0))
        b = bar(v=100.0)
        small = Order(Side.BUY, 1.0, OrderType.MARKET, created_ts=0)
        big = Order(Side.BUY, 10.0, OrderType.MARKET, created_ts=0)
        fs = model.execute_market(small, b, reference_price=100.0, ts=0)
        fb = model.execute_market(big, b, reference_price=100.0, ts=0)
        # 10x the participation => 10x the impact slippage.
        self.assertAlmostEqual(fs.price, 100.0 * (1 + 0.01))   # 1/100
        self.assertAlmostEqual(fb.price, 100.0 * (1 + 0.10))   # 10/100

    def test_taker_fee_charged(self):
        model = ExecutionModel(ExecutionConfig(half_spread=0.0, impact_coefficient=0.0,
                                               taker_fee=0.006, max_participation=1.0))
        b = bar(v=1000.0)
        o = Order(Side.BUY, 2.0, OrderType.MARKET, created_ts=0)
        f = model.execute_market(o, b, reference_price=50.0, ts=0)
        self.assertAlmostEqual(f.fee, 50.0 * 2.0 * 0.006)

    def test_partial_fill_capped_by_participation(self):
        model = ExecutionModel(ExecutionConfig(max_participation=0.1, half_spread=0.0,
                                               impact_coefficient=0.0, taker_fee=0.0))
        b = bar(v=100.0)  # cap = 10 units
        o = Order(Side.BUY, 25.0, OrderType.MARKET, created_ts=0)
        f = model.execute_market(o, b, reference_price=100.0, ts=0)
        self.assertAlmostEqual(f.qty, 10.0)
        self.assertAlmostEqual(o.remaining_qty, 15.0)
        self.assertEqual(o.status.value, "partially_filled")

    def test_all_or_nothing_when_partials_disabled(self):
        model = ExecutionModel(ExecutionConfig(max_participation=0.1,
                                               allow_partial_fills=False))
        b = bar(v=100.0)  # cap = 10
        too_big = Order(Side.BUY, 25.0, OrderType.MARKET, created_ts=0)
        self.assertIsNone(model.execute_market(too_big, b, reference_price=100.0, ts=0))
        fits = Order(Side.BUY, 5.0, OrderType.MARKET, created_ts=0)
        self.assertIsNotNone(model.execute_market(fits, b, reference_price=100.0, ts=0))


class TestLimitExecution(unittest.TestCase):
    def test_buy_limit_only_fills_when_price_reached(self):
        model = ExecutionModel(ExecutionConfig(max_participation=1.0, maker_fee=0.0))
        # Low is 90; a buy limit at 95 should fill, at 80 should not.
        b = bar(o=100.0, h=110.0, l=90.0, c=105.0, v=1000.0)
        hit = Order(Side.BUY, 1.0, OrderType.LIMIT, created_ts=0, limit_price=95.0)
        miss = Order(Side.BUY, 1.0, OrderType.LIMIT, created_ts=0, limit_price=80.0)
        self.assertIsNotNone(model.execute_limit(hit, b, ts=0))
        self.assertIsNone(model.execute_limit(miss, b, ts=0))

    def test_sell_limit_only_fills_when_price_reached(self):
        model = ExecutionModel(ExecutionConfig(max_participation=1.0, maker_fee=0.0))
        b = bar(o=100.0, h=110.0, l=90.0, c=105.0, v=1000.0)
        hit = Order(Side.SELL, 1.0, OrderType.LIMIT, created_ts=0, limit_price=105.0)
        miss = Order(Side.SELL, 1.0, OrderType.LIMIT, created_ts=0, limit_price=120.0)
        self.assertIsNotNone(model.execute_limit(hit, b, ts=0))
        self.assertIsNone(model.execute_limit(miss, b, ts=0))

    def test_gap_through_gives_price_improvement(self):
        model = ExecutionModel(ExecutionConfig(max_participation=1.0, maker_fee=0.0))
        # Open gaps below a buy limit of 95 (opens at 90): fill at 90, not 95.
        b = bar(o=90.0, h=96.0, l=88.0, c=94.0, v=1000.0)
        o = Order(Side.BUY, 1.0, OrderType.LIMIT, created_ts=0, limit_price=95.0)
        f = model.execute_limit(o, b, ts=0)
        self.assertAlmostEqual(f.price, 90.0)
        self.assertEqual(f.liquidity, Liquidity.MAKER)

    def test_limit_fill_at_limit_price_when_no_gap(self):
        model = ExecutionModel(ExecutionConfig(max_participation=1.0, maker_fee=0.0))
        b = bar(o=100.0, h=110.0, l=93.0, c=105.0, v=1000.0)
        o = Order(Side.BUY, 1.0, OrderType.LIMIT, created_ts=0, limit_price=95.0)
        f = model.execute_limit(o, b, ts=0)
        self.assertAlmostEqual(f.price, 95.0)


if __name__ == "__main__":
    unittest.main()
