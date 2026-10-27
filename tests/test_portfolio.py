import unittest

from execsim.models import Fill, Liquidity, Side
from execsim.portfolio import Portfolio


def fill(side, qty, price, fee=0.0, ts=0):
    return Fill(order_id=1, ts=ts, side=side, qty=qty, price=price, fee=fee,
                liquidity=Liquidity.TAKER)


class TestPortfolio(unittest.TestCase):
    def test_cash_and_position_after_buy(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 2.0, 100.0, fee=1.2))
        self.assertAlmostEqual(p.position, 2.0)
        self.assertAlmostEqual(p.cash, 10_000.0 - 200.0 - 1.2)
        self.assertAlmostEqual(p.avg_price, 100.0)

    def test_average_cost_basis(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 1.0, 100.0))
        p.apply_fill(fill(Side.BUY, 1.0, 200.0))
        self.assertAlmostEqual(p.avg_price, 150.0)
        self.assertAlmostEqual(p.position, 2.0)

    def test_realized_pnl_on_close(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 1.0, 100.0, ts=0))
        p.apply_fill(fill(Side.SELL, 1.0, 120.0, ts=10))
        self.assertEqual(len(p.realized_trades), 1)
        t = p.realized_trades[0]
        self.assertAlmostEqual(t.gross_pnl, 20.0)
        self.assertAlmostEqual(t.net_pnl, 20.0)      # no fees here
        self.assertEqual(t.side, Side.BUY)
        self.assertAlmostEqual(p.position, 0.0)

    def test_fees_reduce_net_pnl(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 1.0, 100.0, fee=1.0, ts=0))
        p.apply_fill(fill(Side.SELL, 1.0, 110.0, fee=1.0, ts=10))
        t = p.realized_trades[0]
        self.assertAlmostEqual(t.gross_pnl, 10.0)
        self.assertAlmostEqual(t.fees, 2.0)
        self.assertAlmostEqual(t.net_pnl, 8.0)
        self.assertAlmostEqual(p.total_fees, 2.0)

    def test_partial_close_then_full(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 2.0, 100.0, ts=0))
        p.apply_fill(fill(Side.SELL, 1.0, 110.0, ts=5))   # close half
        self.assertAlmostEqual(p.position, 1.0)
        self.assertEqual(len(p.realized_trades), 1)
        self.assertAlmostEqual(p.realized_trades[0].gross_pnl, 10.0)
        p.apply_fill(fill(Side.SELL, 1.0, 90.0, ts=9))    # close rest at a loss
        self.assertAlmostEqual(p.position, 0.0)
        self.assertAlmostEqual(p.realized_trades[1].gross_pnl, -10.0)

    def test_flip_long_to_short(self):
        p = Portfolio(starting_cash=100_000.0)
        p.apply_fill(fill(Side.BUY, 1.0, 100.0, ts=0))
        p.apply_fill(fill(Side.SELL, 3.0, 120.0, ts=5))   # close 1 long, open 2 short
        self.assertAlmostEqual(p.position, -2.0)
        self.assertAlmostEqual(p.avg_price, 120.0)        # new short basis
        self.assertEqual(len(p.realized_trades), 1)
        self.assertAlmostEqual(p.realized_trades[0].gross_pnl, 20.0)

    def test_equity_marks_to_market(self):
        p = Portfolio(starting_cash=10_000.0)
        p.apply_fill(fill(Side.BUY, 1.0, 100.0))
        self.assertAlmostEqual(p.equity(150.0), 10_000.0 - 100.0 + 150.0)


if __name__ == "__main__":
    unittest.main()
