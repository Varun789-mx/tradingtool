import sys, argparse
import numpy as np, pandas as pd, yfinance as yf

WATCH = ["RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","SBIN","ITC","AXISBANK","LT","TATAMOTORS"]
SLOTS = 75  # 5-min bars, 9:15 to 15:30

def bars(sym):
    df = yf.download(f"{sym}.NS", period="59d", interval="5m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_convert("Asia/Kolkata")
    return [g for _, g in df.groupby(df.index.date)]

def orb_signal(g):
    """First 5-min close above the 15-min opening range -> (bar index, entry, stop, or_hi, or_lo)"""
    hi, lo = g.High.iloc[:3].max(), g.Low.iloc[:3].min()
    for i in range(3, len(g)):
        if g.Close.iloc[i] > hi:
            return i, g.Close.iloc[i], (hi + lo) / 2, hi, lo
    return None

def history_replay(days, r):
    res = {"target": 0, "stop": 0, "open": 0}
    for g in days:
        if len(g) < SLOTS - 3: continue
        s = orb_signal(g)
        if not s: continue
        i, entry, stop, _, _ = s
        risk = entry - stop
        if risk <= 0: continue
        tgt, out = entry + r * risk, "open"
        for j in range(i + 1, min(len(g), 72)):
            if g.Low.iloc[j] <= stop: out = "stop"; break
            if g.High.iloc[j] >= tgt: out = "target"; break
        res[out] += 1
    return res

def simulate(days, k, entry, stop, tgt, n=10000, seed=19):
    full = [g for g in days if len(g) == SLOTS]
    R = np.array([np.log(g.Close.values[1:] / g.Close.values[:-1]) for g in full])  # days x 74
    rng = np.random.default_rng(seed)
    slots = np.arange(k, 71)  # bars after entry until ~15:15
    pick = rng.integers(0, len(R), (n, len(slots)))
    path = entry * np.exp(np.cumsum(R[pick, slots - 1], axis=1))
    hs, ht = path <= stop, path >= tgt
    fs = np.where(hs.any(1), hs.argmax(1), 10**6)
    ft = np.where(ht.any(1), ht.argmax(1), 10**6)
    return (ft < fs).mean(), (fs < ft).mean(), (fs == ft).mean()

def cmd_scan(a):
    print(f"{'STOCK':12}{'ENTRY':>9}{'STOP':>9}{'TARGET':>9}  RR")
    for s in WATCH:
        d = bars(s)
        sig = orb_signal(d[-1])
        if not sig: continue
        i, e, st, hi, lo = sig
        print(f"{s:12}{e:9.2f}{st:9.2f}{e + a.r * (e - st):9.2f}  1:{a.r}")

def cmd_plan(a):
    d = bars(a.symbol); g = d[-1]
    sig = orb_signal(g)
    if not sig: return print("No ORB trigger yet.")
    i, e, st, hi, lo = sig
    risk = e - st; tgt = e + a.r * risk
    qty = int(a.maxloss // risk)
    print(f"Entry {e:.2f} | Stop {st:.2f} | Target {tgt:.2f} | Risk/share {risk:.2f}")
    print(f"Qty {qty} | Capital needed {qty*e:,.0f} | Max loss {qty*risk:,.0f} | Max profit {qty*risk*a.r:,.0f}")
    pt, ps, pn = simulate(d[:-1], i + 1, e, st, tgt)
    print(f"\nTime Machine (10,000 sims): target first {pt:.1%} | stop first {ps:.1%} | neither {pn:.1%}")
    h = history_replay(d[:-1], a.r)
    print(f"Past {sum(h.values())} signals: {h}")
    ev = pt * risk * a.r - ps * risk
    print(f"Expected value/share (before costs): {ev:.2f}  <- if negative or near zero, skip the trade")

p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="c", required=True)
s = sp.add_parser("scan"); s.add_argument("--r", type=float, default=1.5); s.set_defaults(f=cmd_scan)
s = sp.add_parser("plan"); s.add_argument("symbol"); s.add_argument("--r", type=float, default=1.5)
s.add_argument("--maxloss", type=float, default=500); s.set_defaults(f=cmd_plan)
a = p.parse_args(); a.f(a)
