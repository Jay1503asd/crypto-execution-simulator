"""Minimal client for Coinbase's public market-data API (no auth required).

Uses the Coinbase Exchange REST endpoint for historical candles:

    GET https://api.exchange.coinbase.com/products/{product_id}/candles

The endpoint returns rows of ``[time, low, high, open, close, volume]`` (note
the non-OHLC ordering), newest-first, capped at 300 rows per request, so this
client paginates over the requested window. Granularity must be one of the
supported second-values. Only the Python standard library is used.

Reference: Coinbase Exchange API — Get Product Candles.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..models import Candle

# Supported granularities (seconds) -> human label.
GRANULARITIES: Dict[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    21600: "6h",
    86400: "1d",
}

_EXCHANGE_BASE = "https://api.exchange.coinbase.com"
_MAX_ROWS_PER_REQUEST = 300


class CoinbaseError(RuntimeError):
    pass


class CoinbaseClient:
    """Fetches historical candles from Coinbase's public API."""

    def __init__(
        self,
        base_url: str = _EXCHANGE_BASE,
        timeout: float = 20.0,
        max_retries: int = 3,
        user_agent: str = "execsim/0.1 (+https://local)",
        pause_seconds: float = 0.34,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        # Public endpoints are rate limited (~3 req/s); pace requests politely.
        self.pause_seconds = pause_seconds

    # -- public API ---------------------------------------------------------

    def get_products(self) -> List[dict]:
        """Return the list of tradeable products (e.g. BTC-USD, ETH-USD)."""
        return self._get_json("/products")

    def get_candles(
        self,
        product_id: str,
        granularity: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Candle]:
        """Fetch candles for ``product_id`` between ``start`` and ``end``.

        ``start``/``end`` are Unix timestamps (seconds, UTC). If omitted, the
        most recent ``_MAX_ROWS_PER_REQUEST`` bars are returned. Results are
        de-duplicated and returned oldest-first.
        """
        if granularity not in GRANULARITIES:
            raise ValueError(
                f"unsupported granularity {granularity}; "
                f"choose one of {sorted(GRANULARITIES)}"
            )

        if end is None:
            end = int(time.time())
        if start is None:
            start = end - granularity * _MAX_ROWS_PER_REQUEST

        if start >= end:
            raise ValueError("start must be strictly before end")

        window = granularity * _MAX_ROWS_PER_REQUEST
        collected: Dict[int, Candle] = {}

        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + window, end)
            rows = self._get_json(
                f"/products/{urllib.parse.quote(product_id)}/candles",
                params={
                    "granularity": granularity,
                    "start": _iso(chunk_start),
                    "end": _iso(chunk_end),
                },
            )
            for row in rows:
                # row = [time, low, high, open, close, volume]
                ts = int(row[0])
                collected[ts] = Candle(
                    ts=ts,
                    low=float(row[1]),
                    high=float(row[2]),
                    open=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            chunk_start = chunk_end
            if self.pause_seconds:
                time.sleep(self.pause_seconds)

        return [collected[ts] for ts in sorted(collected)]

    # -- internals ----------------------------------------------------------

    def _get_json(self, path: str, params: Optional[dict] = None):
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read().decode("utf-8")
                return json.loads(payload)
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                # 429 (rate limit) / 5xx are transient; back off and retry.
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    last_err = exc
                    time.sleep(self.pause_seconds * attempt * 2)
                    continue
                body = exc.read().decode("utf-8", "replace") if exc.fp else ""
                raise CoinbaseError(f"HTTP {exc.code} for {url}: {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.pause_seconds * attempt * 2)
                    continue
                raise CoinbaseError(f"network error for {url}: {exc}") from exc
        raise CoinbaseError(f"request failed for {url}: {last_err}")


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
