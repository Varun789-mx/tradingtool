from __future__ import annotations

from typing import Any

import pandas as pd


def _blocked(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": "BLOCKED",
        "reason": reason,
        "direction": None,
        "entry": None,
        "stop": None,
        "target": None,
        "partial": None,
        "bar_time": None,
    }


def _waiting(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": "WAITING",
        "reason": reason,
        "direction": None,
        "entry": None,
        "stop": None,
        "target": None,
        "partial": None,
        "bar_time": None,
    }


def _expired(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": "EXPIRED",
        "reason": reason,
        "direction": None,
        "entry": None,
        "stop": None,
        "target": None,
        "partial": None,
        "bar_time": None,
    }


def _triggered(
    name: str,
    direction: str,
    entry: float,
    stop: float,
    r_mult: float,
    partial_r: float,
    bar_time,
    note: str = "",
) -> dict[str, Any]:
    risk = abs(entry - stop)
    if direction == "long":
        target = entry + r_mult * risk
        partial = entry + partial_r * risk
    else:
        target = entry - r_mult * risk
        partial = entry - partial_r * risk
    return {
        "name": name,
        "state": "TRIGGERED",
        "reason": note or "setup triggered on closed bar",
        "direction": direction,
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "partial": float(partial),
        "bar_time": bar_time,
        "risk": float(risk),
    }


def opening_range(closed: pd.DataFrame) -> tuple[float, float] | None:
    if closed is None or len(closed) < 3:
        return None
    first = closed.iloc[:3]
    return float(first["High"].max()), float(first["Low"].min())


def detect_orb(
    closed: pd.DataFrame,
    bias_label: str,
    cfg: dict[str, Any],
    allow: bool,
) -> dict[str, Any]:
    name = "ORB"
    if not allow:
        return _blocked(name, "signals not allowed in this phase")
    if closed is None or closed.empty:
        return _blocked(name, "no closed bars")
    or_ = opening_range(closed)
    if or_ is None:
        return _waiting(name, "building opening range (need 3 closed bars)")
    or_hi, or_lo = or_
    px = float(closed["Close"].iloc[2])
    width_pct = (or_hi - or_lo) / px * 100.0 if px else 0.0
    mn = float(cfg.get("orb", {}).get("min_range_pct", 0.15))
    mx = float(cfg.get("orb", {}).get("max_range_pct", 0.80))
    if width_pct < mn:
        return _blocked(name, f"range too narrow ({width_pct:.2f}% < {mn}%)")
    if width_pct > mx:
        return _blocked(name, f"range too wide ({width_pct:.2f}% > {mx}%)")
    mid = 0.5 * (or_hi + or_lo)
    r_mult = float(cfg["r_multiple_target"])
    partial_r = float(cfg["partial_at_R"])
    want_long = bias_label == "BULLISH"
    want_short = bias_label == "BEARISH"
    if not want_long and not want_short:
        return _blocked(name, "bias NEUTRAL")
    after = closed.iloc[3:]
    if after.empty:
        return _waiting(name, f"OR {or_lo:.0f}-{or_hi:.0f}; waiting for close beyond range")
    for ts, row in after.iterrows():
        close = float(row["Close"])
        sess = float(row.get("session_avg", close))
        if want_long and close > or_hi:
            if close <= sess:
                return _blocked(name, "breakout close not above session average")
            return _triggered(name, "long", close, mid, r_mult, partial_r, ts, "ORB long: close above range")
        if want_short and close < or_lo:
            if close >= sess:
                return _blocked(name, "breakdown close not below session average")
            return _triggered(name, "short", close, mid, r_mult, partial_r, ts, "ORB short: close below range")
        if want_long and close < or_lo:
            return _expired(name, "opposite break first (failed for this bias)")
        if want_short and close > or_hi:
            return _expired(name, "opposite break first (failed for this bias)")
    return _waiting(name, f"OR {or_lo:.0f}-{or_hi:.0f}; no close beyond range yet")


def detect_pullback(
    closed: pd.DataFrame,
    bias_label: str,
    cfg: dict[str, Any],
    allow: bool,
) -> dict[str, Any]:
    name = "PULLBACK"
    if not allow:
        return _blocked(name, "signals not allowed in this phase")
    if closed is None or len(closed) < 8:
        return _waiting(name, "need more bars for EMA trend")
    r_mult = float(cfg["r_multiple_target"])
    partial_r = float(cfg["partial_at_R"])
    ema9 = closed["ema9"]
    ema21 = closed["ema21"]
    want_long = bias_label == "BULLISH"
    want_short = bias_label == "BEARISH"
    if not want_long and not want_short:
        return _blocked(name, "bias NEUTRAL")

    touched = False
    swing = None
    for i in range(5, len(closed)):
        row = closed.iloc[i]
        e9 = float(ema9.iloc[i])
        e21 = float(ema21.iloc[i])
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        sess = float(row.get("session_avg", close))
        atr = float(row.get("atr14", 0.0) or 0.0)
        trend_long = e9 > e21 and close > sess
        trend_short = e9 < e21 and close < sess
        if want_long and trend_long:
            if low <= e21 * 1.001:
                touched = True
                swing = low if swing is None else min(swing, low)
            elif touched and close > e9:
                entry = close
                if atr <= 0:
                    stop = swing if swing is not None else entry * 0.998
                else:
                    swing_risk = entry - float(swing if swing is not None else entry - atr)
                    risk = min(swing_risk, 1.0 * atr)
                    risk = max(risk, 0.5 * atr)
                    stop = entry - risk
                return _triggered(name, "long", entry, stop, r_mult, partial_r, closed.index[i], "pullback reclaim EMA9")
        elif want_short and trend_short:
            if high >= e21 * 0.999:
                touched = True
                swing = high if swing is None else max(swing, high)
            elif touched and close < e9:
                entry = close
                if atr <= 0:
                    stop = swing if swing is not None else entry * 1.002
                else:
                    swing_risk = float(swing if swing is not None else entry + atr) - entry
                    risk = min(swing_risk, 1.0 * atr)
                    risk = max(risk, 0.5 * atr)
                    stop = entry + risk
                return _triggered(name, "short", entry, stop, r_mult, partial_r, closed.index[i], "pullback reject EMA9")
        else:
            touched = False
            swing = None
    return _waiting(name, "trend ok wait for EMA21 touch then EMA9 reclaim" if want_long or want_short else "no trend")


def detect_failed_breakout(
    closed: pd.DataFrame,
    bias_label: str,
    cfg: dict[str, Any],
    allow: bool,
) -> dict[str, Any]:
    name = "FAILED_BO"
    if not allow:
        return _blocked(name, "signals not allowed in this phase")
    if closed is None or closed.empty:
        return _blocked(name, "no closed bars")
    or_ = opening_range(closed)
    if or_ is None:
        return _waiting(name, "need opening range")
    or_hi, or_lo = or_
    r_mult = float(cfg["r_multiple_target"])
    partial_r = float(cfg["partial_at_R"])
    after = closed.iloc[3:]
    if after.empty:
        return _waiting(name, "waiting for a range break that fails")
    closes = after["Close"].astype(float)
    highs = after["High"].astype(float)
    lows = after["Low"].astype(float)
    times = list(after.index)
    n = len(after)
    for i in range(n):
        c = float(closes.iloc[i])
        if c > or_hi:
            extreme = float(highs.iloc[i])
            for j in range(i + 1, min(i + 4, n)):
                extreme = max(extreme, float(highs.iloc[j]))
                if float(closes.iloc[j]) <= or_hi:
                    if bias_label == "BULLISH":
                        return _expired(name, "failed upside break but bias is bullish")
                    entry = float(closes.iloc[j])
                    stop = extreme
                    return _triggered(
                        name,
                        "short",
                        entry,
                        stop,
                        r_mult,
                        partial_r,
                        times[j],
                        "failed breakout short (back inside range)",
                    )
            if i + 3 < n and all(float(closes.iloc[k]) > or_hi for k in range(i, min(i + 4, n))):
                continue
        if c < or_lo:
            extreme = float(lows.iloc[i])
            for j in range(i + 1, min(i + 4, n)):
                extreme = min(extreme, float(lows.iloc[j]))
                if float(closes.iloc[j]) >= or_lo:
                    if bias_label == "BEARISH":
                        return _expired(name, "failed downside break but bias is bearish")
                    entry = float(closes.iloc[j])
                    stop = extreme
                    return _triggered(
                        name,
                        "long",
                        entry,
                        stop,
                        r_mult,
                        partial_r,
                        times[j],
                        "failed breakdown long (back inside range)",
                    )
    return _waiting(name, "no failed breakout yet")


def evaluate_setups(
    closed: pd.DataFrame,
    bias_label: str,
    cfg: dict[str, Any],
    allow: bool,
    stale: bool,
    extra_block: str | None = None,
) -> list[dict[str, Any]]:
    if stale:
        allow = False
        extra_block = extra_block or "STALE data"
    if extra_block and not allow:
        return [
            _blocked("ORB", extra_block),
            _blocked("PULLBACK", extra_block),
            _blocked("FAILED_BO", extra_block),
        ]
    return [
        detect_orb(closed, bias_label, cfg, allow),
        detect_pullback(closed, bias_label, cfg, allow),
        detect_failed_breakout(closed, bias_label, cfg, allow),
    ]
