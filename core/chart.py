from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import IST
from core.signals import opening_range

CHART_DIR = Path(__file__).resolve().parent.parent / "charts"


def _write_html(fig: go.Figure, path: Path) -> Path:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    html = path.read_text(encoding="utf-8")
    html = html.replace("<head>", '<head><meta http-equiv="refresh" content="30">', 1)
    path.write_text(html, encoding="utf-8")
    return path


def render_live_chart(
    index_name: str,
    session: pd.DataFrame,
    plan: dict[str, Any] | None,
    paper: Any | None,
    premium_trace: list[tuple[Any, float]],
    pnl_trace: list[tuple[Any, float]],
) -> Path:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=(f"{index_name} 5m", "Modeled option premium", "Cumulative paper P&L (INR)"),
    )
    if session is not None and not session.empty:
        fig.add_trace(
            go.Candlestick(
                x=session.index,
                open=session["Open"],
                high=session["High"],
                low=session["Low"],
                close=session["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
        if "session_avg" in session.columns:
            fig.add_trace(
                go.Scatter(x=session.index, y=session["session_avg"], name="session avg", line=dict(width=1)),
                row=1,
                col=1,
            )
        if "ema9" in session.columns:
            fig.add_trace(go.Scatter(x=session.index, y=session["ema9"], name="EMA9", line=dict(width=1)), row=1, col=1)
        if "ema21" in session.columns:
            fig.add_trace(go.Scatter(x=session.index, y=session["ema21"], name="EMA21", line=dict(width=1)), row=1, col=1)
        or_ = opening_range(session)
        if or_ is not None:
            lo, hi = or_[1], or_[0]
            x0, x1 = session.index[0], session.index[min(len(session) - 1, 3)]
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=lo,
                y1=hi,
                fillcolor="rgba(180,180,80,0.15)",
                line=dict(width=1),
                row=1,
                col=1,
            )
    if plan:
        for y, name, color in (
            (plan.get("entry_index"), "entry", "blue"),
            (plan.get("stop_index"), "stop", "red"),
            (plan.get("target_index"), "target", "green"),
        ):
            if y is None:
                continue
            fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=name, row=1, col=1)
    if paper is not None:
        fig.add_trace(
            go.Scatter(
                x=[paper.entry_time],
                y=[paper.entry_index],
                mode="markers",
                marker=dict(size=10, symbol="triangle-up"),
                name="paper entry",
            ),
            row=1,
            col=1,
        )
    if premium_trace:
        xs, ys = zip(*premium_trace)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), name="premium"), row=2, col=1)
    if pnl_trace:
        xs, ys = zip(*pnl_trace)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), name="pnl", fill="tozeroy"), row=3, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=True)
    return _write_html(fig, CHART_DIR / f"live_{index_name}.html")


def save_trade_chart(
    index_name: str,
    session: pd.DataFrame,
    plan: dict[str, Any] | None,
    paper: Any | None,
    premium_trace: list,
    pnl_trace: list,
    day: date,
    n: int,
) -> Path:
    path = CHART_DIR / f"trade_{day.isoformat()}_{n}.html"
    live = render_live_chart(index_name, session, plan, paper, premium_trace, pnl_trace)
    html = live.read_text(encoding="utf-8")
    path.write_text(html, encoding="utf-8")
    return path
