# Crypto Execution Simulator & Strategy Backtester

A realistic **execution simulator** for crypto trading strategies, built on
historical Coinbase market data. The goal is not "another AI trading bot" — it
is to model *how orders actually get filled* so you can see whether a strategy
that looks good on paper survives transaction costs and execution constraints.

> A strategy can look profitable on historical prices and still lose money once
> fees, slippage, and partial fills are included. This tool is designed to
> expose that gap.

**Zero dependencies.** Pure Python standard library (3.8+). No pandas, no numpy,
nothing to install.

---

## Why this is more than a notebook backtest

Two design choices do the heavy lifting:

1. **Lookahead-safe by construction.** A strategy only ever receives candles up
   to *and including* the bar it is deciding on. A decision made at the close of
   bar `t` cannot be executed before bar `t+1`. There is no code path by which a
   strategy can read a future candle. This is enforced by the engine and
   [regression-tested](tests/test_lookahead.py).

2. **Explicit execution modeling.** Orders are matched against real OHLCV bars
   with:
   - **Fees** — separate maker/taker rates (Coinbase-like defaults).
   - **Slippage** — a half-spread cost plus size-dependent *market impact* that
     grows with how much of a bar's volume your order consumes.
   - **Partial fills** — a participation cap limits how much of a bar's volume a
     single order can take; the remainder carries into later bars.

The demo runs every strategy twice — once with realistic costs and once in an
idealized cost-free world — so the cost of execution is visible, not hidden.

---

## Quick start

```bash
# Run the end-to-end demo (fetches real Coinbase data, falls back to synthetic)
python examples/run_demo.py

# Force offline synthetic data
python examples/run_demo.py --synthetic

# Run the tests
python -m unittest discover -s tests
```

## Using the CLI

```bash
# 1) Fetch historical candles from Coinbase's public API into a CSV
python -m execsim fetch --product BTC-USD --granularity 3600 --days 180 \
    --out data/btc_1h.csv

# 2) Backtest several strategies against the SAME data, write an HTML report
python -m execsim backtest --csv data/btc_1h.csv \
    --strategy "ma_crossover:fast=20,slow=50" \
    --strategy "mean_reversion:lookback=30,entry_z=1.5,exit_z=0.3" \
    --html report.html --csv-out metrics.csv
# buy-and-hold is always added automatically as the benchmark

# No data file? The backtester generates synthetic candles so it still runs:
python -m execsim backtest --strategy "ma_crossover:fast=10,slow=40"

# Generate a synthetic dataset explicitly
python -m execsim synth --bars 2000 --granularity 3600 --seed 7 --out data/synth.csv
```

### Granularities

Coinbase supports these bar sizes (seconds): `60, 300, 900, 3600, 21600, 86400`
(1m, 5m, 15m, 1h, 6h, 1d).

### Tuning the execution model

```bash
python -m execsim backtest --csv data/btc_1h.csv \
    --strategy "ma_crossover:fast=20,slow=50" \
    --taker-fee 0.006 --maker-fee 0.002 \
    --half-spread 0.0005 --impact 0.5 --max-participation 0.10
```

Set the cost knobs to zero to reproduce the naive, idealized result and compare.

---

## Strategies

| Name | Spec | Parameters |
|------|------|-----------|
| Buy & Hold | `buyhold` | `allocation` |
| MA Crossover | `ma_crossover` | `fast`, `slow`, `allocation`, `allow_short` |
| Mean Reversion | `mean_reversion` | `lookback`, `entry_z`, `exit_z`, `allocation`, `allow_short` |

Add your own by subclassing `execsim.strategies.base.Strategy` and implementing
`on_bar(ctx)`. The `ctx` object is your only window onto the market:

```python
from execsim.strategies.base import Strategy

class MyStrategy(Strategy):
    name = "my_strategy"

    def on_bar(self, ctx):
        closes = ctx.closes()           # prices up to and including now
        if len(closes) < 10:
            return
        if closes[-1] > max(closes[-10:-1]) and ctx.position == 0:
            ctx.target_notional_fraction(0.95, tag="breakout")  # go ~95% long
        elif ctx.position > 0 and closes[-1] < closes[-2]:
            ctx.close_position()
```

Register it in `execsim/strategies/__init__.py`'s `REGISTRY` to use it from the CLI.

---

## Metrics

Reported per strategy, net of modeled costs, with a buy-and-hold comparison:

- Total return, CAGR, final equity
- Annualized volatility, **Sharpe**, **Sortino**, **Calmar**
- **Max drawdown** (and underwater curve in the HTML report)
- **Win rate**, profit factor, expectancy per trade, number of trades
- Total fees paid, market exposure (fraction of time in a position)

Ratios are annualized from the bar granularity, so results are comparable across
timeframes.

---

## How the engine processes each bar

For every bar `i`, in this exact order:

1. **Match** orders submitted on earlier bars against bar `i`. Market orders
   fill at bar `i`'s **open** (plus slippage); resting limit orders fill if bar
   `i`'s range reaches their price (with price improvement on gaps).
2. **Mark to market** at bar `i`'s **close** and record the equity point.
3. **Ask the strategy** for new decisions, passing it `candles[0 .. i]` only.
   New orders become eligible starting at bar `i+1`.

Because step 3 happens after matching and marking, and new orders are not
eligible until the next iteration, a strategy can never trade on information
from inside the bar it is reacting to.

---

## Project layout

```
execsim/
  models.py              Candle, Order, Fill, enums
  data/
    loader.py            CSV load/save, validation, synthetic (GBM) generator
    coinbase.py          public Coinbase candles client (paginated, retrying)
  execution/
    engine.py            fees, slippage, partial fills, market/limit matching
  portfolio.py           cash/position accounting, realized trades, equity curve
  strategies/
    base.py              Strategy ABC + lookahead-safe StrategyContext
    buy_and_hold.py, ma_crossover.py, mean_reversion.py
  backtest.py            lookahead-safe event loop
  metrics.py             returns, drawdown, Sharpe/Sortino, win rate, ...
  report.py              text report + self-contained HTML/SVG report
  cli.py                 fetch / synth / backtest subcommands
examples/run_demo.py     end-to-end demo (costs vs. no-costs)
tests/                   unittest suite incl. lookahead regression tests
```

---

## Notes & limitations

- Fills are modeled at the bar level (OHLCV). Intrabar path is approximated, not
  reconstructed from tick data. This is a deliberate, documented assumption.
- The Coinbase client uses the public, unauthenticated candles endpoint; it is
  rate limited, so large history requests are paginated and paced.
- Defaults approximate retail Coinbase fees. Set your actual tier for accuracy.
- This is a research/simulation tool, not trading or financial advice.
