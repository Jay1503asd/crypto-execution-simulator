#!/usr/bin/env python3
"""End-to-end demo of the execution simulator.

Runs three strategies (moving-average crossover, mean reversion, and the
buy-and-hold benchmark) over the *same* historical data, then prints the
comparison table and — crucially — shows how modeled transaction costs change
the ranking versus a cost-free (idealized) world.

Data source: tries to fetch real Coinbase candles; falls back to deterministic
synthetic data if the network is unavailable, so the demo always runs.

Usage::

    python examples/run_demo.py                 # auto: Coinbase, else synthetic
    python examples/run_demo.py --synthetic     # force synthetic
    python examples/run_demo.py --product ETH-USD --granularity 3600 --days 180
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running directly from the repo without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execsim.backtest import Backtester
from execsim.data.coinbase import CoinbaseClient, CoinbaseError
from execsim.data.loader import generate_synthetic, infer_granularity
from execsim.execution.engine import ExecutionConfig
from execsim.metrics import compute_metrics
from execsim.report import html_report, text_report
from execsim.strategies import BuyAndHold, MeanReversion, MovingAverageCrossover


def load_data(args):
    if not args.synthetic:
        try:
            import time
            client = CoinbaseClient()
            end = int(time.time())
            start = end - args.days * 86400
            print(f"Fetching {args.product} {args.granularity}s candles from Coinbase...",
                  file=sys.stderr)
            candles = client.get_candles(args.product, args.granularity, start, end)
            if len(candles) >= 100:
                print(f"  got {len(candles)} real candles", file=sys.stderr)
                return candles, args.product
            print("  too few candles returned; falling back to synthetic",
                  file=sys.stderr)
        except CoinbaseError as exc:
            print(f"  Coinbase unavailable ({exc}); using synthetic data",
                  file=sys.stderr)

    candles = generate_synthetic(
        n=2000, granularity=args.granularity, start_price=30_000.0,
        annual_drift=0.3, annual_vol=0.7, seed=42,
    )
    return candles, "SYNTH-USD"


def main():
    ap = argparse.ArgumentParser(description="execsim end-to-end demo")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data")
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--granularity", type=int, default=3600)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--html", default="demo_report.html")
    args = ap.parse_args()

    candles, product = load_data(args)
    granularity = infer_granularity(candles)

    def make_strategies():
        return [
            MovingAverageCrossover(fast=20, slow=50),
            MeanReversion(lookback=30, entry_z=1.5, exit_z=0.3),
            BuyAndHold(),  # benchmark
        ]

    # --- realistic (with costs) -------------------------------------------
    realistic = ExecutionConfig()  # Coinbase-like defaults
    bt = Backtester(starting_cash=100_000.0, exec_config=realistic)
    results = [bt.run(s, candles, granularity=granularity) for s in make_strategies()]
    results[-1].strategy_name = "buy_and_hold"  # ensure canonical benchmark name

    print("\n### WITH REALISTIC EXECUTION COSTS "
          "(fees + slippage + partial fills) ###\n")
    print(text_report([compute_metrics(r) for r in results]))

    # --- idealized (no costs) ---------------------------------------------
    free = ExecutionConfig(taker_fee=0.0, maker_fee=0.0, half_spread=0.0,
                           impact_coefficient=0.0, max_participation=1.0)
    bt_free = Backtester(starting_cash=100_000.0, exec_config=free)
    results_free = [bt_free.run(s, candles, granularity=granularity)
                    for s in make_strategies()]
    results_free[-1].strategy_name = "buy_and_hold"

    print("\n\n### IDEALIZED, COST-FREE WORLD (what a naive notebook shows) ###\n")
    print(text_report([compute_metrics(r) for r in results_free]))

    print("\nThe gap between the two tables is the point of this project: "
          "costs and\nexecution assumptions, not the signal, often decide "
          "whether a strategy works.")

    html = html_report(results, product=product)
    with open(args.html, "w") as fh:
        fh.write(html)
    print(f"\nWrote HTML report (realistic costs) to {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
