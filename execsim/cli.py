"""Command-line interface for the execution simulator.

Examples
--------
Fetch 1-hour BTC-USD candles from Coinbase into a CSV::

    python -m execsim fetch --product BTC-USD --granularity 3600 \
        --days 90 --out data/btc_1h.csv

Generate deterministic synthetic data for offline use::

    python -m execsim synth --bars 1500 --granularity 3600 --seed 7 \
        --out data/synth.csv

Backtest several strategies against the same data and write an HTML report::

    python -m execsim backtest --csv data/btc_1h.csv \
        --strategy buyhold \
        --strategy "ma_crossover:fast=20,slow=50" \
        --strategy "mean_reversion:lookback=30,entry_z=1.5" \
        --html report.html
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Sequence

from .backtest import Backtester
from .data.coinbase import GRANULARITIES, CoinbaseClient, CoinbaseError
from .data.loader import (
    generate_synthetic,
    infer_granularity,
    load_csv,
    save_csv,
    validate_candles,
)
from .execution.engine import ExecutionConfig
from .metrics import compute_metrics
from .models import Candle
from .report import html_report, text_report
from .strategies import REGISTRY


# ---------------------------------------------------------------------------
# strategy spec parsing  ("name:param=value,param=value")
# ---------------------------------------------------------------------------

def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in value or "e" in low:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_strategy_spec(spec: str):
    """Build a strategy instance from ``name:k=v,k=v``."""
    name, _, param_str = spec.partition(":")
    name = name.strip()
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown strategy '{name}'. available: {', '.join(sorted(REGISTRY))}"
        )
    kwargs: Dict[str, object] = {}
    if param_str.strip():
        for pair in param_str.split(","):
            if not pair.strip():
                continue
            key, _, raw = pair.partition("=")
            if not raw:
                raise SystemExit(f"bad parameter '{pair}' in '{spec}' (expected k=v)")
            kwargs[key.strip()] = _coerce(raw.strip())
    try:
        return REGISTRY[name](**kwargs)
    except TypeError as exc:
        raise SystemExit(f"invalid parameters for '{name}': {exc}")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> int:
    end = int(time.time())
    if args.days is not None:
        start = end - int(args.days * 86400)
    else:
        start = None
    client = CoinbaseClient()
    try:
        print(f"Fetching {args.product} @ {GRANULARITIES.get(args.granularity, args.granularity)} "
              f"from Coinbase ...", file=sys.stderr)
        candles = client.get_candles(args.product, args.granularity, start=start, end=end)
    except CoinbaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not candles:
        print("error: no candles returned", file=sys.stderr)
        return 2
    _warn(candles)
    save_csv(candles, args.out)
    print(f"Saved {len(candles)} candles to {args.out}", file=sys.stderr)
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    candles = generate_synthetic(
        n=args.bars,
        granularity=args.granularity,
        start_price=args.start_price,
        annual_drift=args.drift,
        annual_vol=args.vol,
        seed=args.seed,
    )
    save_csv(candles, args.out)
    print(f"Saved {len(candles)} synthetic candles to {args.out}", file=sys.stderr)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    candles = _load_candles(args)
    if candles is None:
        return 2
    _warn(candles)
    granularity = args.granularity or infer_granularity(candles)

    exec_config = ExecutionConfig(
        taker_fee=args.taker_fee,
        maker_fee=args.maker_fee,
        half_spread=args.half_spread,
        impact_coefficient=args.impact,
        max_participation=args.max_participation,
    )
    bt = Backtester(
        starting_cash=args.cash,
        exec_config=exec_config,
        liquidate_at_end=not args.no_liquidate,
    )

    specs: List[str] = list(args.strategy or [])
    # Always include buy-and-hold as the benchmark unless the user already did.
    if not any(s.split(":")[0].strip() == "buyhold" for s in specs):
        specs.append("buyhold")

    results = []
    for spec in specs:
        strat = parse_strategy_spec(spec)
        # Give the benchmark the canonical name the report looks for.
        if spec.split(":")[0].strip() == "buyhold":
            strat.name = "buy_and_hold"
        result = bt.run(strat, candles, granularity=granularity)
        results.append(result)

    metrics = [compute_metrics(r) for r in results]
    print(text_report(metrics))

    if args.html:
        html = html_report(results, product=args.product or "")
        with open(args.html, "w") as fh:
            fh.write(html)
        print(f"\nWrote HTML report to {args.html}", file=sys.stderr)

    if args.csv_out:
        _write_metrics_csv(metrics, args.csv_out)
        print(f"Wrote metrics CSV to {args.csv_out}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_candles(args: argparse.Namespace) -> Optional[List[Candle]]:
    if args.csv:
        try:
            return load_csv(args.csv)
        except (OSError, ValueError) as exc:
            print(f"error loading {args.csv}: {exc}", file=sys.stderr)
            return None
    # No CSV: synthesize so the tool is usable offline.
    print("no --csv provided; using synthetic data (pass --csv for real data)",
          file=sys.stderr)
    return generate_synthetic(
        n=args.synth_bars, granularity=args.granularity or 3600, seed=args.synth_seed
    )


def _warn(candles: Sequence[Candle]) -> None:
    warnings = validate_candles(list(candles))
    if warnings:
        print(f"data quality: {len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings[:10]:
            print(f"  - {w}", file=sys.stderr)
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more", file=sys.stderr)


def _write_metrics_csv(metrics, path: str) -> None:
    import csv

    if not metrics:
        return
    keys = list(metrics[0].as_dict().keys())
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for m in metrics:
            d = m.as_dict()
            writer.writerow([d[k] for k in keys])


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="execsim",
        description="Crypto execution simulator & strategy backtester",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fetch
    f = sub.add_parser("fetch", help="download historical candles from Coinbase")
    f.add_argument("--product", default="BTC-USD", help="e.g. BTC-USD, ETH-USD")
    f.add_argument("--granularity", type=int, default=3600, choices=sorted(GRANULARITIES))
    f.add_argument("--days", type=float, default=90, help="how many days back to fetch")
    f.add_argument("--out", required=True, help="output CSV path")
    f.set_defaults(func=cmd_fetch)

    # synth
    s = sub.add_parser("synth", help="generate synthetic candles (offline)")
    s.add_argument("--bars", type=int, default=1500)
    s.add_argument("--granularity", type=int, default=3600, choices=sorted(GRANULARITIES))
    s.add_argument("--start-price", type=float, default=20000.0)
    s.add_argument("--drift", type=float, default=0.25, help="annualized drift")
    s.add_argument("--vol", type=float, default=0.80, help="annualized volatility")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_synth)

    # backtest
    b = sub.add_parser("backtest", help="run one or more strategies on the same data")
    b.add_argument("--csv", help="input candle CSV (omit to use synthetic data)")
    b.add_argument("--granularity", type=int, default=None,
                   help="override granularity (seconds); inferred if omitted")
    b.add_argument("--strategy", action="append",
                   help="strategy spec 'name:k=v,...'; repeatable. "
                        f"names: {', '.join(sorted(REGISTRY))}")
    b.add_argument("--cash", type=float, default=100_000.0, help="starting cash")
    b.add_argument("--taker-fee", type=float, default=0.006)
    b.add_argument("--maker-fee", type=float, default=0.002)
    b.add_argument("--half-spread", type=float, default=0.0005)
    b.add_argument("--impact", type=float, default=0.5, help="market-impact coefficient")
    b.add_argument("--max-participation", type=float, default=0.10,
                   help="max fraction of a bar's volume one order may fill")
    b.add_argument("--no-liquidate", action="store_true",
                   help="do not force-close the open position at the end")
    b.add_argument("--product", default="", help="label for the report header")
    b.add_argument("--html", help="write a self-contained HTML report to this path")
    b.add_argument("--csv-out", help="write the metrics table to this CSV path")
    b.add_argument("--synth-bars", type=int, default=1500,
                   help="bars to synthesize when no --csv is given")
    b.add_argument("--synth-seed", type=int, default=7)
    b.set_defaults(func=cmd_backtest)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
