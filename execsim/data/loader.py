"""Load, save, validate and synthesize OHLCV candle data.

CSV schema (header required): ``ts,open,high,low,close,volume``
where ``ts`` is the bar open time as a Unix timestamp in seconds (UTC).
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import List, Optional

from ..models import Candle

CSV_HEADER = ["ts", "open", "high", "low", "close", "volume"]


def save_csv(candles: List[Candle], path: str) -> None:
    """Write candles to a CSV file (creating parent directories as needed)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for c in candles:
            writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])


def load_csv(path: str) -> List[Candle]:
    """Read candles from a CSV file and return them sorted by timestamp."""
    candles: List[Candle] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [col for col in CSV_HEADER if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV {path!r} is missing required columns: {missing}. "
                f"Expected header: {','.join(CSV_HEADER)}"
            )
        for row in reader:
            candles.append(
                Candle(
                    ts=int(float(row["ts"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    candles.sort(key=lambda c: c.ts)
    return candles


def validate_candles(candles: List[Candle]) -> List[str]:
    """Return a list of human-readable data-quality warnings (empty == clean).

    Checks ordering, duplicate timestamps, OHLC consistency, non-negative
    volume, and irregular bar spacing.
    """
    warnings: List[str] = []
    if not candles:
        return ["dataset is empty"]

    prev_ts: Optional[int] = None
    spacings = {}
    for i, c in enumerate(candles):
        if c.high < c.low:
            warnings.append(f"bar {i} (ts={c.ts}): high < low")
        hi = max(c.open, c.close)
        lo = min(c.open, c.close)
        if c.high < hi - 1e-9:
            warnings.append(f"bar {i} (ts={c.ts}): high below open/close")
        if c.low > lo + 1e-9:
            warnings.append(f"bar {i} (ts={c.ts}): low above open/close")
        if c.volume < 0:
            warnings.append(f"bar {i} (ts={c.ts}): negative volume")
        if min(c.open, c.high, c.low, c.close) <= 0:
            warnings.append(f"bar {i} (ts={c.ts}): non-positive price")
        if prev_ts is not None:
            if c.ts == prev_ts:
                warnings.append(f"bar {i} (ts={c.ts}): duplicate timestamp")
            elif c.ts < prev_ts:
                warnings.append(f"bar {i} (ts={c.ts}): timestamps out of order")
            else:
                spacings[c.ts - prev_ts] = spacings.get(c.ts - prev_ts, 0) + 1
        prev_ts = c.ts

    if len(spacings) > 1:
        # Report the dominant spacing and how many bars deviate from it.
        dominant = max(spacings, key=spacings.get)
        irregular = sum(n for s, n in spacings.items() if s != dominant)
        warnings.append(
            f"irregular bar spacing: {irregular} gaps differ from the dominant "
            f"{dominant}s interval (possible missing bars)"
        )
    return warnings


def infer_granularity(candles: List[Candle]) -> int:
    """Infer the dominant bar interval (seconds) from the data."""
    if len(candles) < 2:
        return 60
    spacings = {}
    for a, b in zip(candles, candles[1:]):
        d = b.ts - a.ts
        if d > 0:
            spacings[d] = spacings.get(d, 0) + 1
    if not spacings:
        return 60
    return max(spacings, key=spacings.get)


def generate_synthetic(
    n: int = 1000,
    start_ts: int = 1_600_000_000,
    granularity: int = 3600,
    start_price: float = 20_000.0,
    annual_drift: float = 0.25,
    annual_vol: float = 0.80,
    seed: Optional[int] = None,
    base_volume: float = 50.0,
) -> List[Candle]:
    """Generate synthetic OHLCV candles via geometric Brownian motion.

    Useful for offline development and deterministic tests (pass ``seed``).
    Intrabar high/low are derived from a per-bar volatility so the OHLC
    relationships are always internally consistent.
    """
    rng = random.Random(seed)
    periods_per_year = (365 * 24 * 3600) / granularity
    dt = 1.0 / periods_per_year
    mu = annual_drift
    sigma = annual_vol

    candles: List[Candle] = []
    price = start_price
    for i in range(n):
        ts = start_ts + i * granularity
        open_p = price
        # GBM step for the close.
        shock = rng.gauss(0.0, 1.0)
        close_p = open_p * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * shock)
        # Intrabar range scaled by per-bar volatility.
        bar_sigma = sigma * math.sqrt(dt)
        spread = abs(open_p) * bar_sigma * (0.5 + rng.random())
        hi = max(open_p, close_p) + spread * rng.random()
        lo = min(open_p, close_p) - spread * rng.random()
        lo = max(lo, 0.01)
        vol = base_volume * (0.4 + 1.2 * rng.random())
        candles.append(
            Candle(ts=ts, open=open_p, high=hi, low=lo, close=close_p, volume=vol)
        )
        price = close_p
    return candles
