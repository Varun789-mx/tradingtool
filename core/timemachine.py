from __future__ import annotations

import json
import math
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data import IST, CACHE_DIR, session_slice, with_indicators
from core.options import round_trip_costs_np
from core.signals import evaluate_setups

_erf = np.vectorize(math.erf, otypes=[float])
SQRT_2 = math.sqrt(2.0)


def _norm_cdf_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / SQRT_2))


def bs_price_np(
    spot: np.ndarray,
    strike: float,
    t: np.ndarray,
    rate: float,
    sigma: float,
    option_type: str,
) -> np.ndarray:
    spot = np.maximum(spot, 1e-8)
    t = np.asarray(t, dtype=float)
    price = np.empty_like(spot, dtype=float)
    expired = t <= 1e-12
    if option_type == "call":
        price[expired] = np.maximum(spot[expired] - strike, 0.0)
    else:
        price[expired] = np.maximum(strike - spot[expired], 0.0)
    live = ~expired
    if not np.any(live):
        return price
    s = spot[live]
    tt = t[live]
    vol_sqrt = sigma * np.sqrt(tt)
    vol_sqrt = np.where(vol_sqrt < 1e-12, 1e-12, vol_sqrt)
    d1 = (np.log(s / strike) + (rate + 0.5 * sigma * sigma) * tt) / vol_sqrt
    d2 = d1 - vol_sqrt
    df = np.exp(-rate * tt)
    if option_type == "call":
        price[live] = s * _norm_cdf_np(d1) - strike * df * _norm_cdf_np(d2)
    else:
        price[live] = strike * df * _norm_cdf_np(-d2) - s * _norm_cdf_np(-d1)
    return price


def _tod_slot(ts: pd.Timestamp) -> int:
    t = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)
    minutes = t.hour * 60 + t.minute
    return (minutes - (9 * 60 + 15)) // 5


def build_slot_returns(history: pd.DataFrame) -> np.ndarray:
    """Shape (n_days, 75) log returns by 5m slot. Empty slots are nan."""
    if history is None or history.empty:
        return np.empty((0, 75))
    hist = session_slice(history)
    rows = []
    for _, g in hist.groupby(hist.index.date):
        g = g.sort_index()
        slot_ret = np.full(75, np.nan)
        closes = g["Close"].astype(float)
        for i in range(1, len(g)):
            sl = _tod_slot(g.index[i])
            if 0 <= sl < 75:
                a, b = float(closes.iloc[i - 1]), float(closes.iloc[i])
                if a > 0 and b > 0:
                    slot_ret[sl] = math.log(b / a)
        rows.append(slot_ret)
    if not rows:
        return np.empty((0, 75))
    return np.vstack(rows)


def remaining_slots(now: datetime) -> np.ndarray:
    now = now.astimezone(IST)
    end = now.replace(hour=15, minute=15, second=0, microsecond=0)
    if now >= end:
        return np.array([], dtype=int)
    cur_min = now.hour * 60 + now.minute
    start_slot = max(0, (cur_min - (9 * 60 + 15)) // 5 + 1)
    last_slot = (15 * 60 + 10 - (9 * 60 + 15)) // 5  # 15:10 bar, covers to 15:15
    return np.arange(start_slot, last_slot + 1, dtype=int)


def monte_carlo(
    history: pd.DataFrame,
    spot: float,
    plan: dict[str, Any],
    now: datetime,
    n_paths: int = 10000,
    seed: int = 42,
    costs_cfg: dict[str, Any] | None = None,
    exchange: str = "NSE",
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    R = build_slot_returns(history)
    slots = remaining_slots(now)
    empty = {
        "n_paths": n_paths,
        "target_first": 0.0,
        "stop_first": 0.0,
        "neither": 1.0,
        "median_pnl": 0.0,
        "worst_5": 0.0,
        "best_5": 0.0,
        "ev": 0.0,
        "verdict": "SKIP",
        "reason": "not enough path data",
        "label": "SKIP",
    }
    if R.size == 0 or slots.size == 0 or plan.get("skip") or plan.get("qty", 0) <= 0:
        if plan.get("skip"):
            empty["reason"] = plan.get("skip_reason") or "position skipped"
        elif slots.size == 0:
            empty["reason"] = "no time left to 15:15"
        return empty
    valid_days = ~np.isnan(R).all(axis=1)
    R = R[valid_days]
    if len(R) < 5:
        empty["reason"] = f"only {len(R)} history days"
        return empty
    for j, sl in enumerate(slots):
        col = R[:, sl]
        if np.all(np.isnan(col)):
            med = np.nanmedian(R)
            R[:, sl] = 0.0 if np.isnan(med) else med
        else:
            fill = np.nanmedian(col)
            nan_m = np.isnan(R[:, sl])
            R[nan_m, sl] = fill
    n_days, _ = R.shape
    n_steps = len(slots)
    pick = rng.integers(0, n_days, size=(n_paths, n_steps))
    rets = R[pick, slots[None, :]]
    path = spot * np.exp(np.cumsum(rets, axis=1))
    strike = float(plan["strike"])
    rate = float(plan.get("rate") or 0.065)
    sigma = float(plan["iv"])
    opt = plan["option_type"]
    t0 = float(plan["t_years"])
    step_years = 5.0 / (365.25 * 24.0 * 60.0)
    t_arr = np.maximum(t0 - step_years * np.arange(1, n_steps + 1), 0.0)
    t_mat = np.broadcast_to(t_arr, path.shape)
    prem = bs_price_np(path, strike, t_mat, rate, sigma, opt)
    stop_p = float(plan["stop_premium"])
    tgt_p = float(plan["target_premium"])
    entry_p = float(plan["entry_premium"])
    qty = int(plan["qty"])
    hit_tgt = prem >= tgt_p
    hit_stp = prem <= stop_p
    inf = n_steps + 1
    ft = np.where(hit_tgt.any(axis=1), hit_tgt.argmax(axis=1), inf)
    fs = np.where(hit_stp.any(axis=1), hit_stp.argmax(axis=1), inf)
    target_first = ft < fs
    stop_first = fs < ft
    neither = ft == fs
    exit_px = np.empty(n_paths)
    exit_px[target_first] = tgt_p
    exit_px[stop_first] = stop_p
    last = prem[:, -1]
    exit_px[neither] = last[neither]
    gross = (exit_px - entry_p) * qty
    costs = round_trip_costs_np(entry_p, exit_px, qty, costs_cfg or {}, exchange)
    net = gross - costs
    tf = float(target_first.mean())
    sf = float(stop_first.mean())
    neu = float(neither.mean())
    ev = float(net.mean())
    take = ev > 0 and tf > sf
    if take:
        verdict = (
            f"TAKE: sample EV after costs is ₹{ev:,.0f} with target-first {tf:.0%} vs stop-first {sf:.0%}. "
            "This is a bootstrap from past same-time returns, not a prediction."
        )
        label = "TAKE"
    else:
        verdict = (
            f"SKIP: EV ₹{ev:,.0f}, target-first {tf:.0%} vs stop-first {sf:.0%}. "
            "Probabilities from resampled history; they are uncertain."
        )
        label = "SKIP"
    return {
        "n_paths": n_paths,
        "n_days": int(n_days),
        "target_first": tf,
        "stop_first": sf,
        "neither": neu,
        "median_pnl": float(np.median(net)),
        "worst_5": float(np.quantile(net, 0.05)),
        "best_5": float(np.quantile(net, 0.95)),
        "ev": ev,
        "verdict": verdict,
        "label": label,
        "reason": verdict,
    }


def visible_bars(df: pd.DataFrame, asof: datetime) -> pd.DataFrame:
    """Bars with timestamp <= asof. Callers must not pass future bars."""
    if df is None or df.empty:
        return pd.DataFrame()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    asof = asof.astimezone(IST)
    return df.loc[idx <= asof].copy()


def replay_signal_at(df: pd.DataFrame, asof: datetime, bias_label: str, cfg: dict[str, Any], allow: bool) -> list[dict[str, Any]]:
    vis = visible_bars(df, asof)
    from core.data import closed_and_forming

    closed, _ = closed_and_forming(vis, asof)
    return evaluate_setups(closed, bias_label, cfg, allow, stale=False)


HIST_STATS_PATH = CACHE_DIR / "setup_expectancy.json"


def history_setup_stats(
    history: pd.DataFrame,
    cfg: dict[str, Any],
    index_name: str,
    now: datetime,
    use_cache: bool = True,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{index_name}"
    if use_cache and HIST_STATS_PATH.exists():
        try:
            cached = json.loads(HIST_STATS_PATH.read_text(encoding="utf-8"))
            if cached.get("date") == now.astimezone(IST).date().isoformat() and key in cached.get("by_index", {}):
                return cached["by_index"][key]
        except (OSError, json.JSONDecodeError):
            cached = {"by_index": {}}
    else:
        cached = {"by_index": {}}
    sess = session_slice(with_indicators(history) if history is not None else pd.DataFrame())
    by_setup: dict[str, dict[str, Any]] = {
        n: {"fires": 0, "target": 0, "stop": 0, "neither": 0, "nets": []}
        for n in ("ORB", "PULLBACK", "FAILED_BO")
    }
    if sess.empty:
        return {"setups": by_setup, "n_days": 0}
    days = list(sess.groupby(sess.index.date))
    r_mult = float(cfg["r_multiple_target"])
    for day, g in days:
        g = g.sort_index()
        if len(g) < 20:
            continue
        closed_full = g
        for name, detect in (
            ("ORB", "ORB"),
            ("PULLBACK", "PULLBACK"),
            ("FAILED_BO", "FAILED_BO"),
        ):
            for bias in ("BULLISH", "BEARISH"):
                from core.signals import detect_orb, detect_pullback, detect_failed_breakout

                fn = {"ORB": detect_orb, "PULLBACK": detect_pullback, "FAILED_BO": detect_failed_breakout}[name]
                sig = fn(closed_full, bias, cfg, True)
                if sig.get("state") != "TRIGGERED":
                    continue
                entry_i = sig["bar_time"]
                try:
                    loc = closed_full.index.get_loc(entry_i)
                except KeyError:
                    continue
                if isinstance(loc, slice):
                    loc = loc.start
                if isinstance(loc, np.ndarray):
                    loc = int(loc.flat[0])
                loc = int(loc)
                future = closed_full.iloc[loc + 1 :]
                # cutoff 15:15
                future = future[future.index.tz_convert(IST).time <= dtime(15, 15)]
                entry, stop, target = sig["entry"], sig["stop"], sig["target"]
                outcome = "neither"
                for _, row in future.iterrows():
                    if sig["direction"] == "long":
                        if float(row["Low"]) <= stop:
                            outcome = "stop"
                            break
                        if float(row["High"]) >= target:
                            outcome = "target"
                            break
                    else:
                        if float(row["High"]) >= stop:
                            outcome = "stop"
                            break
                        if float(row["Low"]) <= target:
                            outcome = "target"
                            break
                st = by_setup[name]
                st["fires"] += 1
                st[outcome] += 1
                risk = abs(entry - stop)
                if outcome == "target":
                    st["nets"].append(r_mult)
                elif outcome == "stop":
                    st["nets"].append(-1.0)
                else:
                    st["nets"].append(0.0)
                break  # one fire per setup per day
    out = {}
    n_days = len(days)
    for name, st in by_setup.items():
        nets = st["nets"]
        n = len(nets)
        wr = (st["target"] / n) if n else 0.0
        avg = float(np.mean(nets)) if n else 0.0
        warn = n < 30
        out[name] = {
            "fires": st["fires"],
            "hit_target": st["target"],
            "hit_stop": st["stop"],
            "neither": st["neither"],
            "win_rate": wr,
            "avg_net_R": avg,
            "n": n,
            "n_days": n_days,
            "small_sample": warn,
            "positive_expectancy": avg > 0 and n >= 1,
        }
    payload = {
        "date": now.astimezone(IST).date().isoformat(),
        "by_index": {**cached.get("by_index", {}), key: {"setups": out, "n_days": n_days}},
    }
    if use_cache:
        try:
            HIST_STATS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
    return payload["by_index"][key]


def replay_clock_start(day: date) -> datetime:
    return datetime.combine(day, dtime(8, 0), tzinfo=IST)


def advance_replay(sim: datetime, speed: float, wall_dt: float) -> datetime:
    add = wall_dt * max(speed, 0.1)
    return sim + timedelta(seconds=add)


def slice_no_lookahead(full_day: pd.DataFrame, sim_now: datetime) -> pd.DataFrame:
    """Proof point: evaluation data is strictly timestamp <= sim_now."""
    return visible_bars(full_day, sim_now)
