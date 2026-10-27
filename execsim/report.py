"""Reporting: a terminal-friendly text summary and a self-contained HTML report.

The HTML report embeds inline SVG charts (equity curves normalized to a base of
100, and underwater/drawdown curves) so it opens in any browser with no assets,
no JavaScript, and no third-party libraries.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from .backtest import BacktestResult
from .metrics import Metrics, compute_metrics

# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _num(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:,.2f}"


def _ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


_ROWS: List[Tuple[str, str]] = [
    ("Total return", "total_return_pct"),
    ("CAGR", "cagr_pct"),
    ("Final equity", "final_equity_num"),
    ("Volatility (ann.)", "volatility_annual_pct"),
    ("Sharpe", "sharpe_num"),
    ("Sortino", "sortino_num"),
    ("Max drawdown", "max_drawdown_pct"),
    ("Calmar", "calmar_num"),
    ("Trades", "num_trades_int"),
    ("Win rate", "win_rate_pct"),
    ("Profit factor", "profit_factor_num"),
    ("Expectancy/trade", "expectancy_num"),
    ("Total fees", "total_fees_num"),
    ("Exposure", "exposure_pct"),
]


def _cell(m: Metrics, key: str) -> str:
    field, _, kind = key.rpartition("_")
    value = getattr(m, field)
    if kind == "pct":
        return _pct(value)
    if kind == "int":
        return str(int(value))
    return _num(value)


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------

def text_report(metrics: Sequence[Metrics], benchmark_name: str = "buy_and_hold") -> str:
    if not metrics:
        return "(no results)"

    names = [m.strategy_name for m in metrics]
    label_w = max(len(lbl) for lbl, _ in _ROWS) + 1
    col_w = max(14, max(len(n) for n in names) + 2)

    lines: List[str] = []
    m0 = metrics[0]
    lines.append("=" * (label_w + col_w * len(metrics)))
    lines.append("CRYPTO EXECUTION SIMULATOR — BACKTEST REPORT")
    lines.append(
        f"Period: {_ts(m0.start_ts)} -> {_ts(m0.end_ts)} UTC  |  "
        f"{m0.bars} bars  |  ~{m0.periods_per_year:,.0f} periods/yr"
    )
    lines.append("=" * (label_w + col_w * len(metrics)))

    header = " " * label_w + "".join(n.rjust(col_w) for n in names)
    lines.append(header)
    lines.append("-" * len(header))

    for label, key in _ROWS:
        row = label.ljust(label_w)
        for m in metrics:
            row += _cell(m, key).rjust(col_w)
        lines.append(row)

    # Benchmark comparison block.
    bench = next((m for m in metrics if m.strategy_name == benchmark_name), None)
    if bench is not None and len(metrics) > 1:
        lines.append("-" * len(header))
        lines.append("VS BUY-AND-HOLD")
        row = "Excess return".ljust(label_w)
        for m in metrics:
            excess = m.total_return - bench.total_return
            cell = "-" if m.strategy_name == benchmark_name else _pct(excess)
            row += cell.rjust(col_w)
        lines.append(row)
        row = "Beat benchmark?".ljust(label_w)
        for m in metrics:
            if m.strategy_name == benchmark_name:
                cell = "-"
            else:
                cell = "yes" if m.total_return > bench.total_return else "no"
            row += cell.rjust(col_w)
        lines.append(row)

    lines.append("=" * (label_w + col_w * len(metrics)))
    lines.append(
        "Note: returns are net of modeled fees, slippage, and partial fills. "
        "A strategy can beat buy-and-hold gross yet lose after costs."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG charts (no dependencies)
# ---------------------------------------------------------------------------

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _normalized_equity(result: BacktestResult) -> List[Tuple[int, float]]:
    curve = result.equity_curve
    base = curve[0].equity or 1.0
    return [(pt.ts, pt.equity / base * 100.0) for pt in curve]


def _drawdown_series(result: BacktestResult) -> List[Tuple[int, float]]:
    curve = result.equity_curve
    peak = curve[0].equity
    out = []
    for pt in curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity / peak - 1.0) * 100.0 if peak > 0 else 0.0
        out.append((pt.ts, dd))
    return out


def _svg_line_chart(
    series: List[Tuple[str, List[Tuple[int, float]], str]],
    width: int = 820,
    height: int = 300,
    title: str = "",
    y_suffix: str = "",
    zero_baseline: bool = False,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 16, 28, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_x = [x for _, pts, _ in series for x, _ in pts]
    all_y = [y for _, pts, _ in series for _, y in pts]
    if not all_x or not all_y:
        return f'<svg width="{width}" height="{height}"></svg>'

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if zero_baseline:
        ymax = max(ymax, 0.0)
        ymin = min(ymin, 0.0)
    if ymax == ymin:
        ymax += 1.0
        ymin -= 1.0
    xspan = (xmax - xmin) or 1

    def sx(x: int) -> float:
        return pad_l + (x - xmin) / xspan * plot_w

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    parts: List[str] = [f'<svg width="{width}" height="{height}" '
                        f'viewBox="0 0 {width} {height}" '
                        f'xmlns="http://www.w3.org/2000/svg" font-family="monospace">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        parts.append(f'<text x="{pad_l}" y="18" font-size="13" fill="#111">'
                    f'{html.escape(title)}</text>')

    # gridlines + y labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yval = ymax - frac * (ymax - ymin)
        y = pad_t + frac * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                    f'y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3:.1f}" font-size="10" '
                    f'text-anchor="end" fill="#666">{yval:,.1f}{y_suffix}</text>')

    if zero_baseline and ymin < 0 < ymax:
        yz = sy(0.0)
        parts.append(f'<line x1="{pad_l}" y1="{yz:.1f}" x2="{width - pad_r}" '
                    f'y2="{yz:.1f}" stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>')

    # x labels (start / mid / end)
    for frac in (0.0, 0.5, 1.0):
        xval = int(xmin + frac * xspan)
        x = pad_l + frac * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" font-size="10" '
                    f'text-anchor="middle" fill="#666">{_ts(xval)}</text>')

    for name, pts, color in series:
        if not pts:
            continue
        if zero_baseline:
            # Filled underwater area.
            d = [f'M {sx(pts[0][0]):.1f} {sy(0.0):.1f}']
            for x, y in pts:
                d.append(f'L {sx(x):.1f} {sy(y):.1f}')
            d.append(f'L {sx(pts[-1][0]):.1f} {sy(0.0):.1f} Z')
            parts.append(f'<path d="{" ".join(d)}" fill="{color}" '
                        f'fill-opacity="0.18" stroke="none"/>')
        d = [f'M {sx(pts[0][0]):.1f} {sy(pts[0][1]):.1f}']
        for x, y in pts[1:]:
            d.append(f'L {sx(x):.1f} {sy(y):.1f}')
        parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
                    f'stroke-width="1.6"/>')

    # legend
    lx = pad_l + 4
    ly = pad_t + 12
    for i, (name, _, color) in enumerate(series):
        yy = ly + i * 15
        parts.append(f'<rect x="{lx}" y="{yy - 8}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{yy}" font-size="11" fill="#111">'
                    f'{html.escape(name)}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def html_report(
    results: Sequence[BacktestResult],
    title: str = "Crypto Execution Simulator — Backtest Report",
    product: str = "",
) -> str:
    metrics = [compute_metrics(r) for r in results]
    colors = {r.strategy_name: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(results)}

    equity_series = [
        (r.strategy_name, _normalized_equity(r), colors[r.strategy_name]) for r in results
    ]
    dd_series = [
        (r.strategy_name, _drawdown_series(r), colors[r.strategy_name]) for r in results
    ]

    equity_svg = _svg_line_chart(
        equity_series, title="Equity curve (indexed to 100)", y_suffix="")
    dd_svg = _svg_line_chart(
        dd_series, title="Drawdown (%)", y_suffix="%", zero_baseline=True)

    m0 = metrics[0]
    header_meta = (
        f"{html.escape(product) + ' | ' if product else ''}"
        f"{_ts(m0.start_ts)} → {_ts(m0.end_ts)} UTC &middot; {m0.bars} bars"
    )

    # metrics table
    th = "".join(f"<th>{html.escape(m.strategy_name)}</th>" for m in metrics)
    rows_html = []
    for label, key in _ROWS:
        tds = "".join(f"<td>{html.escape(_cell(m, key))}</td>" for m in metrics)
        rows_html.append(f"<tr><th class='rowlabel'>{html.escape(label)}</th>{tds}</tr>")

    bench = next((m for m in metrics if "buy_and_hold" in m.strategy_name
                  or m.strategy_name == "buy_and_hold"), None)
    comp_rows = ""
    if bench is not None and len(metrics) > 1:
        excess_tds = ""
        beat_tds = ""
        for m in metrics:
            if m is bench:
                excess_tds += "<td>—</td>"
                beat_tds += "<td>—</td>"
            else:
                ex = m.total_return - bench.total_return
                cls = "pos" if ex >= 0 else "neg"
                excess_tds += f"<td class='{cls}'>{_pct(ex)}</td>"
                beat = m.total_return > bench.total_return
                beat_tds += (f"<td class='{'pos' if beat else 'neg'}'>"
                            f"{'yes' if beat else 'no'}</td>")
        comp_rows = (
            f"<tr><th class='rowlabel'>Excess vs B&amp;H</th>{excess_tds}</tr>"
            f"<tr><th class='rowlabel'>Beat B&amp;H?</th>{beat_tds}</tr>"
        )

    style = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0;background:#f5f6f8;color:#1a1a1a}
    .wrap{max-width:900px;margin:0 auto;padding:24px}
    h1{font-size:20px;margin:0 0 4px}
    .meta{color:#666;font-size:13px;margin-bottom:20px}
    .card{background:#fff;border:1px solid #e3e5e9;border-radius:10px;
          padding:16px;margin-bottom:20px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eef0f3}
    th{color:#333}
    thead th{border-bottom:2px solid #d7dae0}
    .rowlabel{text-align:left;color:#555;font-weight:600}
    td.pos{color:#16a34a;font-weight:600}
    td.neg{color:#dc2626;font-weight:600}
    .note{color:#777;font-size:12px;margin-top:8px}
    svg{max-width:100%;height:auto;display:block}
    """

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{style}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="meta">{header_meta}</div>
<div class="card">{equity_svg}</div>
<div class="card">{dd_svg}</div>
<div class="card">
<table>
<thead><tr><th class="rowlabel">Metric</th>{th}</tr></thead>
<tbody>
{''.join(rows_html)}
{comp_rows}
</tbody>
</table>
<div class="note">Returns are net of modeled transaction fees, slippage, and
volume-constrained partial fills. Decisions are made at each bar's close and
executed no earlier than the next bar, so no future information is used.</div>
</div>
</div></body></html>"""
