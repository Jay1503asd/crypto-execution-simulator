"""Performance and risk metrics computed from a :class:`BacktestResult`.

All ratios are annualized using the bar granularity so results are comparable
across timeframes. Return-based statistics (Sharpe, Sortino, volatility, max
drawdown) are derived from the mark-to-market *equity curve*; trade-based
statistics (win rate, profit factor, average win/loss) come from realized
round-trip trades.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import List, Optional, Sequence

from .backtest import BacktestResult
from .portfolio import EquityPoint

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class DrawdownInfo:
    max_drawdown: float          # fraction, e.g. 0.25 == -25%
    peak_ts: int
    trough_ts: int
    recovery_ts: Optional[int]   # None if never recovered by end of series
    longest_drawdown_bars: int


@dataclass
class Metrics:
    strategy_name: str
    start_ts: int
    end_ts: int
    bars: int
    periods_per_year: float

    starting_equity: float
    final_equity: float
    total_return: float          # fraction
    cagr: float                  # annualized compound growth, fraction

    volatility_annual: float
    sharpe: float
    sortino: float
    max_drawdown: float          # fraction (positive number)
    calmar: float

    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float            # average net pnl per trade
    total_fees: float
    exposure: float              # fraction of bars holding a non-zero position

    def as_dict(self) -> dict:
        return asdict(self)


def _period_returns(curve: Sequence[EquityPoint]) -> List[float]:
    """Simple per-bar returns of the equity curve."""
    rets: List[float] = []
    for prev, cur in zip(curve, curve[1:]):
        if prev.equity <= 0:
            rets.append(0.0)
        else:
            rets.append(cur.equity / prev.equity - 1.0)
    return rets


def _max_drawdown(curve: Sequence[EquityPoint]) -> DrawdownInfo:
    peak = curve[0].equity
    peak_ts = curve[0].ts
    max_dd = 0.0
    dd_peak_ts = peak_ts
    dd_trough_ts = peak_ts
    recovery_ts: Optional[int] = None

    # Longest drawdown measured in bars from peak until a new high is made.
    longest = 0
    cur_len = 0
    running_peak = curve[0].equity
    for pt in curve:
        if pt.equity >= running_peak:
            running_peak = pt.equity
            cur_len = 0
        else:
            cur_len += 1
            longest = max(longest, cur_len)

        if pt.equity > peak:
            peak = pt.equity
            peak_ts = pt.ts
        dd = (peak - pt.equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            dd_peak_ts = peak_ts
            dd_trough_ts = pt.ts
            recovery_ts = None

    # Did equity recover to the pre-drawdown peak after the trough?
    trough_peak = None
    for pt in curve:
        if pt.ts == dd_peak_ts:
            trough_peak = pt.equity
        if trough_peak is not None and pt.ts > dd_trough_ts and pt.equity >= trough_peak:
            recovery_ts = pt.ts
            break

    return DrawdownInfo(
        max_drawdown=max_dd,
        peak_ts=dd_peak_ts,
        trough_ts=dd_trough_ts,
        recovery_ts=recovery_ts,
        longest_drawdown_bars=longest,
    )


def compute_metrics(result: BacktestResult) -> Metrics:
    curve = result.equity_curve
    if len(curve) < 2:
        raise ValueError("equity curve too short to compute metrics")

    periods_per_year = SECONDS_PER_YEAR / result.granularity
    start_equity = result.starting_cash
    final_equity = curve[-1].equity

    total_return = final_equity / start_equity - 1.0

    # CAGR from elapsed wall-clock time.
    elapsed = max(1, curve[-1].ts - curve[0].ts)
    years = elapsed / SECONDS_PER_YEAR
    if years > 0 and final_equity > 0:
        cagr = (final_equity / start_equity) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    rets = _period_returns(curve)
    mean_r = fmean(rets) if rets else 0.0
    std_r = pstdev(rets) if len(rets) > 1 else 0.0
    downside = [r for r in rets if r < 0]
    downside_std = pstdev(downside) if len(downside) > 1 else 0.0

    vol_annual = std_r * math.sqrt(periods_per_year)
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0
    sortino = (mean_r / downside_std) * math.sqrt(periods_per_year) if downside_std > 0 else 0.0

    dd = _max_drawdown(curve)
    calmar = (cagr / dd.max_drawdown) if dd.max_drawdown > 0 else 0.0

    # Trade-based stats.
    trades = result.realized_trades
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    num_trades = len(trades)
    win_rate = len(wins) / num_trades if num_trades else 0.0
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    avg_win = fmean(wins) if wins else 0.0
    avg_loss = fmean(losses) if losses else 0.0
    expectancy = fmean([t.net_pnl for t in trades]) if trades else 0.0

    exposed_bars = sum(1 for pt in curve if abs(pt.position) > 1e-12)
    exposure = exposed_bars / len(curve)

    return Metrics(
        strategy_name=result.strategy_name,
        start_ts=curve[0].ts,
        end_ts=curve[-1].ts,
        bars=len(curve),
        periods_per_year=periods_per_year,
        starting_equity=start_equity,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        volatility_annual=vol_annual,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=dd.max_drawdown,
        calmar=calmar,
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        total_fees=result.total_fees,
        exposure=exposure,
    )


@dataclass
class Comparison:
    """A strategy's metrics alongside the buy-and-hold benchmark."""

    strategy: Metrics
    benchmark: Metrics

    @property
    def excess_return(self) -> float:
        return self.strategy.total_return - self.benchmark.total_return

    @property
    def beat_benchmark(self) -> bool:
        return self.strategy.total_return > self.benchmark.total_return
