from __future__ import annotations

import math
from datetime import date, datetime, time as dtime, timedelta
from typing import Any

import numpy as np

from core.data import IST, is_trading_day, previous_trading_day

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(spot: float, strike: float, t: float, rate: float, sigma: float) -> tuple[float, float]:
    vol_sqrt = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def years_to_expiry(now: datetime, expiry: date) -> float:
    now = now.astimezone(IST)
    end = datetime.combine(expiry, dtime(15, 30), tzinfo=IST)
    sec = (end - now).total_seconds()
    return max(0.0, sec / (365.25 * 24.0 * 3600.0))


def bs_price(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
) -> float:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t <= 1e-12 or sigma <= 1e-12:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, t, rate, sigma)
    df = math.exp(-rate * t)
    if option_type == "call":
        return spot * norm_cdf(d1) - strike * df * norm_cdf(d2)
    if option_type == "put":
        return strike * df * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError(f"unknown option_type {option_type}")


def bs_greeks(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
) -> dict[str, float]:
    price = bs_price(spot, strike, t, rate, sigma, option_type)
    if t <= 1e-12 or sigma <= 1e-12:
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {"price": price, "delta": delta, "theta_day": 0.0}
    d1, d2 = _d1_d2(spot, strike, t, rate, sigma)
    df = math.exp(-rate * t)
    if option_type == "call":
        delta = norm_cdf(d1)
        theta_yr = (
            -spot * norm_pdf(d1) * sigma / (2.0 * math.sqrt(t))
            - rate * strike * df * norm_cdf(d2)
        )
    else:
        delta = -norm_cdf(-d1)
        theta_yr = (
            -spot * norm_pdf(d1) * sigma / (2.0 * math.sqrt(t))
            + rate * strike * df * norm_cdf(-d2)
        )
    return {"price": price, "delta": delta, "theta_day": theta_yr / 365.25}


def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    option_type: str,
    lo: float = 1e-4,
    hi: float = 5.0,
    steps: int = 80,
) -> float:
    if market_price <= 0:
        raise ValueError("market premium must be positive")
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if market_price < intrinsic * 0.999:
        return lo
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        px = bs_price(spot, strike, t, rate, mid, option_type)
        if px > market_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def round_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def pick_strike(spot: float, step: int, option_type: str, t: float, rate: float, sigma: float) -> int:
    atm = round_strike(spot, step)
    itm = atm - step if option_type == "call" else atm + step
    if itm <= 0:
        itm = atm
    best = atm
    best_err = 1e9
    for k in (atm, itm):
        g = bs_greeks(spot, float(k), max(t, 1e-6), rate, max(sigma, 1e-4), option_type)
        d = abs(g["delta"])
        if 0.5 <= d <= 0.6:
            err = abs(d - 0.55)
        else:
            err = abs(d - 0.55) + 0.5
        if err < best_err:
            best_err = err
            best = k
    return int(best)


def _shift_expiry(d: date, holidays: set[date]) -> date:
    return previous_trading_day(d, holidays)


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def next_weekday_on_or_after(d: date, weekday: int) -> date:
    add = (weekday - d.weekday()) % 7
    return d + timedelta(days=add)


def _still_live(exp: date, now: datetime) -> bool:
    d = now.astimezone(IST).date()
    cutoff = datetime.combine(d, dtime(15, 30), tzinfo=IST)
    if exp > d:
        return True
    return exp == d and now <= cutoff


def nearest_expiry(now: datetime, rule: str, holidays: set[date]) -> date:
    now = now.astimezone(IST)
    d = now.date()
    if rule in ("weekly_tuesday", "weekly_thursday"):
        wd = 1 if rule == "weekly_tuesday" else 3
        for add in range(0, 28):
            raw = next_weekday_on_or_after(d + timedelta(days=add), wd)
            exp = _shift_expiry(raw, holidays)
            if _still_live(exp, now):
                return exp
        raise RuntimeError("could not resolve weekly expiry")
    if rule == "monthly_last_tuesday":
        year, month = d.year, d.month
        for _ in range(4):
            exp = _shift_expiry(last_weekday_of_month(year, month, 1), holidays)
            if _still_live(exp, now):
                return exp
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise RuntimeError("could not resolve monthly expiry")
    raise ValueError(f"unknown expiry rule {rule}")


def txn_pct(costs: dict[str, Any], exchange: str) -> float:
    if str(exchange).upper() == "BSE":
        return float(costs.get("exchange_txn_bse_pct", 0.0325))
    return float(costs.get("exchange_txn_nse_pct", 0.03503))


def round_trip_costs(
    premium_in: float,
    premium_out: float,
    qty: int,
    costs: dict[str, Any],
    exchange: str,
    n_entry_orders: int = 1,
    n_exit_orders: int = 1,
) -> dict[str, float]:
    if qty <= 0:
        return {"total": 0.0, "brokerage": 0.0, "stt": 0.0, "txn": 0.0, "sebi": 0.0, "stamp": 0.0, "gst": 0.0, "slippage": 0.0}
    buy_notional = premium_in * qty
    sell_notional = premium_out * qty
    brokerage = float(costs["brokerage_per_order"]) * (n_entry_orders + n_exit_orders)
    stt = float(costs["stt_sell_premium_pct"]) / 100.0 * sell_notional
    tp = txn_pct(costs, exchange) / 100.0
    txn = tp * (buy_notional + sell_notional)
    sebi = float(costs["sebi_fee_pct"]) / 100.0 * (buy_notional + sell_notional)
    stamp = float(costs["stamp_duty_buy_pct"]) / 100.0 * buy_notional
    gst = float(costs["gst_pct"]) / 100.0 * (brokerage + txn + sebi)
    slip = float(costs["slippage_per_unit"]) * qty * 2.0
    total = brokerage + stt + txn + sebi + stamp + gst + slip
    return {
        "total": total,
        "brokerage": brokerage,
        "stt": stt,
        "txn": txn,
        "sebi": sebi,
        "stamp": stamp,
        "gst": gst,
        "slippage": slip,
    }


def round_trip_costs_np(
    premium_in: float,
    premium_out: np.ndarray,
    qty: int,
    costs: dict[str, Any],
    exchange: str,
    n_entry_orders: int = 1,
    n_exit_orders: int = 1,
) -> np.ndarray:
    buy = premium_in * qty
    sell = premium_out * qty
    brokerage = float(costs["brokerage_per_order"]) * (n_entry_orders + n_exit_orders)
    stt = float(costs["stt_sell_premium_pct"]) / 100.0 * sell
    tp = txn_pct(costs, exchange) / 100.0
    txn = tp * (buy + sell)
    sebi = float(costs["sebi_fee_pct"]) / 100.0 * (buy + sell)
    stamp = float(costs["stamp_duty_buy_pct"]) / 100.0 * buy
    gst = float(costs["gst_pct"]) / 100.0 * (brokerage + txn + sebi)
    slip = float(costs["slippage_per_unit"]) * qty * 2.0
    return brokerage + stt + txn + sebi + stamp + gst + slip


def size_lots(max_loss: float, premium_risk_per_unit: float, lot: int) -> int:
    if premium_risk_per_unit <= 0 or lot <= 0 or max_loss <= 0:
        return 0
    per_lot_risk = premium_risk_per_unit * lot
    return int(math.floor(max_loss / per_lot_risk))


def index_to_premium(
    index_level: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
) -> float:
    return bs_price(index_level, strike, t, rate, sigma, option_type)


def breakeven_index(
    entry_prem: float,
    qty: int,
    costs_fn,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
    spot: float,
) -> float | None:
    """Index level where net P&L after costs is ~0 (long option)."""
    if qty <= 0:
        return None
    lo = spot * 0.8
    hi = spot * 1.2
    if option_type == "put":
        lo, hi = spot * 0.8, spot * 1.05
    else:
        lo, hi = spot * 0.95, spot * 1.2

    def net_at(s: float) -> float:
        px = bs_price(max(s, 1e-6), strike, t, rate, sigma, option_type)
        c = costs_fn(entry_prem, px, qty)
        return (px - entry_prem) * qty - c["total"]

    n_lo, n_hi = net_at(lo), net_at(hi)
    if n_lo > 0 and n_hi > 0:
        return spot
    if n_lo * n_hi > 0:
        # expand
        lo, hi = spot * 0.5, spot * 1.5
        n_lo, n_hi = net_at(lo), net_at(hi)
        if n_lo * n_hi > 0:
            return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        n_mid = net_at(mid)
        if n_lo * n_mid <= 0:
            hi = mid
            n_hi = n_mid
        else:
            lo = mid
            n_lo = n_mid
    return 0.5 * (lo + hi)


def build_plan(
    cfg: dict[str, Any],
    index_name: str,
    spot: float,
    direction: str,
    entry_index: float,
    stop_index: float,
    target_index: float,
    now: datetime,
    iv: float,
    live_premium: float | None = None,
) -> dict[str, Any]:
    meta = cfg["indices"][index_name]
    lot = int(meta["lot"])
    step = int(meta["step"])
    exchange = str(meta["exchange"])
    holidays = {date.fromisoformat(h) for h in cfg.get("holidays", [])}
    expiry = nearest_expiry(now, str(meta["expiry"]), holidays)
    t = years_to_expiry(now, expiry)
    rate = float(cfg["risk_free"])
    option_type = "call" if direction == "long" else "put"
    strike = pick_strike(spot, step, option_type, t, rate, iv)
    g = bs_greeks(entry_index, float(strike), max(t, 1e-8), rate, iv, option_type)
    modeled = g["price"]
    entry_prem = float(live_premium) if live_premium is not None else modeled
    stop_prem = index_to_premium(stop_index, float(strike), max(t, 1e-8), rate, iv, option_type)
    target_prem = index_to_premium(target_index, float(strike), max(t, 1e-8), rate, iv, option_type)
    risk_unit = entry_prem - stop_prem
    max_loss_cash = float(cfg["capital"]) * float(cfg["max_loss_pct_per_trade"]) / 100.0
    lots = size_lots(max_loss_cash, risk_unit, lot) if risk_unit > 0 else 0
    skip = lots < 1
    qty = lots * lot
    capital_needed = entry_prem * qty if qty else 0.0
    max_loss = risk_unit * qty if qty else 0.0
    max_profit = (target_prem - entry_prem) * qty if qty else 0.0
    costs_tgt = round_trip_costs(entry_prem, target_prem, qty, cfg["costs"], exchange) if qty else {"total": 0.0}
    costs_stp = round_trip_costs(entry_prem, stop_prem, qty, cfg["costs"], exchange) if qty else {"total": 0.0}
    net_profit = max_profit - float(costs_tgt["total"])
    net_loss = max_loss + float(costs_stp["total"])
    gross_rr = (target_prem - entry_prem) / risk_unit if risk_unit > 0 else 0.0
    net_rr = net_profit / net_loss if net_loss > 0 else 0.0

    def cfn(pin, pout, q):
        return round_trip_costs(pin, pout, q, cfg["costs"], exchange)

    be = breakeven_index(entry_prem, qty, cfn, float(strike), max(t, 1e-8), rate, iv, option_type, spot) if qty else None
    right = "CE" if option_type == "call" else "PE"
    contract = f"{index_name} {strike} {right} {expiry.isoformat()}"
    expiry_afternoon = expiry == now.astimezone(IST).date() and now.astimezone(IST).time() >= dtime(13, 0)
    return {
        "contract": contract,
        "index": index_name,
        "direction": direction,
        "option_type": option_type,
        "strike": strike,
        "expiry": expiry,
        "t_years": t,
        "iv": iv,
        "delta": g["delta"],
        "theta_day": g["theta_day"],
        "spot": spot,
        "entry_index": entry_index,
        "stop_index": stop_index,
        "target_index": target_index,
        "entry_premium": entry_prem,
        "modeled_premium": modeled,
        "live_premium": live_premium,
        "stop_premium": stop_prem,
        "target_premium": target_prem,
        "lots": lots,
        "qty": qty,
        "lot_size": lot,
        "capital_needed": capital_needed,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "gross_rr": gross_rr,
        "net_rr": net_rr,
        "costs_to_target": float(costs_tgt["total"]),
        "costs_to_stop": float(costs_stp["total"]),
        "net_profit": net_profit,
        "skip": skip,
        "skip_reason": "SKIP: too small for your risk limit" if skip else "",
        "breakeven_index": be,
        "exchange": exchange,
        "expiry_afternoon_warn": expiry_afternoon,
        "stt_flag": bool(cfg["costs"].get("stt_changed_recently")),
    }
