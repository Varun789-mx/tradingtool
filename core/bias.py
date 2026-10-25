from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def _chg(rows: list[dict[str, Any]], ticker: str) -> float | None:
    for r in rows:
        if r.get("ticker") == ticker:
            return r.get("chg_pct")
    return None


def _clip_unit(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    v = x / scale
    return max(-1.0, min(1.0, v))


def compute_bias(
    cfg: dict[str, Any],
    globals_rows: list[dict[str, Any]],
    vix_df: pd.DataFrame,
    session: pd.DataFrame,
    gap_pct: float | None,
    news_sc: float,
    now: datetime,
    holidays: set[date],
    stale: bool,
    day_halt: bool,
) -> dict[str, Any]:
    w = cfg.get("bias_weights", {})
    parts: dict[str, float] = {}

    us = []
    for t in ("ES=F", "NQ=F", "^GSPC", "^IXIC"):
        v = _chg(globals_rows, t)
        if v is not None:
            us.append(v)
    us_avg = sum(us) / len(us) if us else 0.0
    parts["us_futures"] = _clip_unit(us_avg, 0.8) * float(w.get("us_futures", 15))

    asia = []
    for t in ("^N225", "^HSI"):
        v = _chg(globals_rows, t)
        if v is not None:
            asia.append(v)
    asia_avg = sum(asia) / len(asia) if asia else 0.0
    parts["asia"] = _clip_unit(asia_avg, 1.0) * float(w.get("asia", 10))

    crude = _chg(globals_rows, "CL=F") or 0.0
    parts["crude"] = _clip_unit(-crude, 1.5) * float(w.get("crude", 8))

    usdinr = _chg(globals_rows, "USDINR=X") or 0.0
    parts["usdinr"] = _clip_unit(-usdinr, 0.4) * float(w.get("usdinr", 8))

    dxy = _chg(globals_rows, "DX-Y.NYB") or 0.0
    parts["dxy"] = _clip_unit(-dxy, 0.4) * float(w.get("dxy", 5))

    vix_last = None
    vix_chg = None
    if vix_df is not None and not vix_df.empty:
        vix_last = float(vix_df["Close"].iloc[-1])
        if len(vix_df) >= 2:
            prev = float(vix_df["Close"].iloc[-2])
            if prev:
                vix_chg = (vix_last - prev) / prev * 100.0
        parts["vix_level"] = _clip_unit(16.0 - vix_last, 8.0) * float(w.get("vix_level", 10))
        parts["vix_change"] = _clip_unit(-(vix_chg or 0.0), 4.0) * float(w.get("vix_change", 8))
    else:
        parts["vix_level"] = 0.0
        parts["vix_change"] = 0.0

    parts["gap"] = _clip_unit(gap_pct or 0.0, 0.6) * float(w.get("gap", 12))
    parts["news"] = _clip_unit(news_sc, 50.0) * float(w.get("news", 12))

    if session is not None and not session.empty:
        last = float(session["Close"].iloc[-1])
        avg = float(session["session_avg"].iloc[-1]) if "session_avg" in session.columns else last
        dist = (last - avg) / avg * 100.0 if avg else 0.0
        parts["intraday_trend"] = _clip_unit(dist, 0.25) * float(w.get("intraday_trend", 12))
    else:
        parts["intraday_trend"] = 0.0

    score = sum(parts.values())
    score = max(-100.0, min(100.0, score))
    if score >= 15:
        label = "BULLISH"
    elif score <= -15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    factors: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        factors.append({"name": name, "ok": ok, "detail": detail})

    add("data_fresh", not stale, "fresh" if not stale else "STALE — signals blocked")
    add("daily_limits", not day_halt, "ok" if not day_halt else "STOP TRADING — limit/goal/count")
    add("bias_not_neutral", label != "NEUTRAL", label)
    if vix_last is not None:
        add("vix_not_extreme", vix_last < 25, f"VIX {vix_last:.2f}")
    else:
        add("vix_not_extreme", False, "VIX missing")
    add("us_futures", us_avg >= -0.4, f"US avg {us_avg:+.2f}%")
    add("asia", asia_avg >= -0.5, f"Asia avg {asia_avg:+.2f}%")
    add("usd_inr", usdinr <= 0.35, f"USDINR {usdinr:+.2f}%")
    add("news_not_panic", news_sc > -40, f"news {news_sc:+.0f} (bias input, not a forecast)")
    if session is not None and not session.empty and "session_avg" in session.columns:
        last = float(session["Close"].iloc[-1])
        avg = float(session["session_avg"].iloc[-1])
        add("vs_session_avg", True, f"spot {last:.0f} vs avg {avg:.0f}")
    else:
        add("vs_session_avg", False, "no session bars")

    today = now.date()
    event_hits = [e for e in cfg.get("events", []) if str(e.get("date")) == today.isoformat()]
    expiry_likely = now.weekday() in (1, 3)
    add("event_flag", not event_hits, ", ".join(e.get("name", "") for e in event_hits) or "no listed event")
    add("weekday_expiry_watch", True, "Tue/Thu expiry watch" if expiry_likely else "not typical weekly expiry")

    go = all(f["ok"] for f in factors if f["name"] in ("data_fresh", "daily_limits"))
    return {
        "score": score,
        "label": label,
        "parts": parts,
        "factors": factors,
        "go": go,
        "vix": vix_last,
        "vix_chg": vix_chg,
        "events": event_hits,
        "news_note": "News score is a bias input, never a forecast.",
    }


def what_changed(prev: dict[str, Any] | None, cur: dict[str, Any]) -> str:
    if not prev:
        return f"bias init {cur['label']} {cur['score']:+.0f}"
    ds = cur["score"] - prev["score"]
    arrow = "▲" if ds > 1 else "▼" if ds < -1 else "→"
    flips = []
    if prev.get("label") != cur.get("label"):
        flips.append(f"{prev.get('label')}→{cur.get('label')}")
    old_parts = prev.get("parts") or {}
    new_parts = cur.get("parts") or {}
    biggest = None
    biggest_abs = 0.0
    for k, v in new_parts.items():
        dv = float(v) - float(old_parts.get(k, 0.0))
        if abs(dv) > biggest_abs:
            biggest_abs = abs(dv)
            biggest = (k, dv)
    bit = f"{biggest[0]} {biggest[1]:+.1f}" if biggest and biggest_abs >= 1 else "quiet"
    extra = (" " + " ".join(flips)) if flips else ""
    return f"{arrow} {ds:+.1f} ({bit}){extra}"
