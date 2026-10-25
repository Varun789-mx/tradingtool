from __future__ import annotations

import math
import time
import unittest
from datetime import date, datetime, time as dtime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.bias import compute_bias
from core.config import DEFAULT_CONFIG
from core.data import IST, add_session_features
from core.options import (
    bs_price,
    implied_vol,
    nearest_expiry,
    round_trip_costs,
    size_lots,
)
from core.signals import detect_failed_breakout, detect_orb, detect_pullback
from core.timemachine import monte_carlo, slice_no_lookahead, visible_bars
from core.journal import PaperPosition
from trade import App

CFG = DEFAULT_CONFIG


def _bars(closes, highs=None, lows=None, start=None) -> pd.DataFrame:
    n = len(closes)
    start = start or datetime(2026, 10, 21, 9, 15, tzinfo=IST)
    idx = pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(n)])
    c = np.array(closes, dtype=float)
    h = np.array(highs if highs is not None else c + 2, dtype=float)
    l = np.array(lows if lows is not None else c - 2, dtype=float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 0.0}, index=idx)
    return add_session_features(df)


class TestBlackScholes(unittest.TestCase):
    def test_put_call_parity(self):
        s, k, t, r, sig = 24500.0, 24500.0, 3 / 365.25, 0.065, 0.14
        c = bs_price(s, k, t, r, sig, "call")
        p = bs_price(s, k, t, r, sig, "put")
        rhs = s - k * math.exp(-r * t)
        self.assertAlmostEqual(c - p, rhs, places=6)

    def test_iv_round_trip(self):
        s, k, t, r, sig = 24600.0, 24550.0, 5 / 365.25, 0.065, 0.18
        px = bs_price(s, k, t, r, sig, "call")
        iv = implied_vol(px, s, k, t, r, "call")
        self.assertAlmostEqual(iv, sig, places=3)


class TestSizing(unittest.TestCase):
    def test_lots_and_skip(self):
        self.assertEqual(size_lots(2000, 10, 65), 3)
        self.assertEqual(size_lots(200, 10, 65), 0)
        self.assertEqual(size_lots(650, 10, 65), 1)


class TestCosts(unittest.TestCase):
    def test_round_trip_positive(self):
        c = round_trip_costs(100.0, 120.0, 65, CFG["costs"], "NSE")
        self.assertGreater(c["total"], 40.0)
        self.assertGreater(c["stt"], 0)
        self.assertGreater(c["gst"], 0)
        bse = round_trip_costs(100.0, 120.0, 20, CFG["costs"], "BSE")
        nse = round_trip_costs(100.0, 120.0, 20, CFG["costs"], "NSE")
        self.assertLess(bse["txn"], nse["txn"])


class TestExpiry(unittest.TestCase):
    def test_holiday_shift_tuesday(self):
        holidays = {date(2026, 10, 20), date(2026, 11, 10), date(2026, 11, 24)}
        now = datetime(2026, 10, 19, 10, 0, tzinfo=IST)
        exp = nearest_expiry(now, "weekly_tuesday", holidays)
        self.assertEqual(exp, date(2026, 10, 19))
        now2 = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
        self.assertEqual(nearest_expiry(now2, "weekly_tuesday", holidays), date(2026, 10, 27))
        now3 = datetime(2026, 10, 15, 10, 0, tzinfo=IST)
        self.assertEqual(nearest_expiry(now3, "monthly_last_tuesday", holidays), date(2026, 10, 27))
        now4 = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
        self.assertEqual(nearest_expiry(now4, "weekly_thursday", holidays), date(2026, 10, 22))


class TestSetups(unittest.TestCase):
    def test_orb_long(self):
        closes = [24020, 24010, 23990, 24120]
        highs = [24080, 24040, 24020, 24140]
        lows = [23990, 23980, 23940, 24090]
        df = _bars(closes, highs, lows)
        sig = detect_orb(df, "BULLISH", CFG, True)
        self.assertEqual(sig["state"], "TRIGGERED")
        self.assertEqual(sig["direction"], "long")
        self.assertAlmostEqual(sig["stop"], 0.5 * (24080 + 23940))

    def test_pullback_long(self):
        n = 12
        start = datetime(2026, 10, 21, 9, 15, tzinfo=IST)
        idx = pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(n)])
        close = np.linspace(100, 110, n)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 0.0,
                "session_avg": close - 2,
                "ema9": close + 0.2,
                "ema21": close - 0.8,
                "atr14": np.full(n, 2.0),
            },
            index=idx,
        )
        # bar 8 touches ema21, bar 9 closes back above ema9
        df.iloc[8, df.columns.get_loc("Low")] = float(df["ema21"].iloc[8]) - 0.1
        df.iloc[8, df.columns.get_loc("Close")] = float(df["ema9"].iloc[8]) - 0.3
        df.iloc[9, df.columns.get_loc("Close")] = float(df["ema9"].iloc[9]) + 0.5
        df.iloc[9, df.columns.get_loc("High")] = float(df.iloc[9]["Close"]) + 0.2
        df.iloc[9, df.columns.get_loc("Low")] = float(df["ema21"].iloc[9]) + 0.4
        sig = detect_pullback(df, "BULLISH", CFG, True)
        self.assertEqual(sig["state"], "TRIGGERED")
        self.assertEqual(sig["direction"], "long")

    def test_failed_breakout_short(self):
        closes = [24020, 24010, 23990, 24120, 24050]
        highs = [24080, 24040, 24020, 24150, 24090]
        lows = [23990, 23980, 23940, 24090, 24020]
        df = _bars(closes, highs, lows)
        sig = detect_failed_breakout(df, "BEARISH", CFG, True)
        self.assertEqual(sig["state"], "TRIGGERED")
        self.assertEqual(sig["direction"], "short")


class TestBias(unittest.TestCase):
    def test_bounds(self):
        now = datetime(2026, 10, 21, 11, 0, tzinfo=IST)
        vix = pd.DataFrame(
            {"Close": [12.0, 11.5]},
            index=pd.DatetimeIndex([now - timedelta(minutes=5), now], tz=IST),
        )
        sess = _bars([24000, 24010, 24020, 24040])
        out = compute_bias(
            CFG,
            [
                {"ticker": "ES=F", "chg_pct": 50.0},
                {"ticker": "NQ=F", "chg_pct": 50.0},
                {"ticker": "^N225", "chg_pct": 50.0},
                {"ticker": "^HSI", "chg_pct": 50.0},
                {"ticker": "CL=F", "chg_pct": -50.0},
                {"ticker": "USDINR=X", "chg_pct": -50.0},
                {"ticker": "DX-Y.NYB", "chg_pct": -50.0},
            ],
            vix,
            sess,
            5.0,
            100.0,
            now,
            set(),
            False,
            False,
        )
        self.assertGreaterEqual(out["score"], -100)
        self.assertLessEqual(out["score"], 100)
        self.assertIn(out["label"], ("BULLISH", "BEARISH", "NEUTRAL"))


class TestMonteCarlo(unittest.TestCase):
    def test_deterministic_seed(self):
        day0 = date(2026, 9, 1)
        frames = []
        rng = np.random.default_rng(0)
        for d in range(40):
            dt = day0 + timedelta(days=d)
            if dt.weekday() >= 5:
                continue
            start = datetime.combine(dt, dtime(9, 15), tzinfo=IST)
            px = 24000.0
            rows = []
            for i in range(75):
                px *= math.exp(float(rng.normal(0, 0.0008)))
                ts = start + timedelta(minutes=5 * i)
                rows.append((ts, px))
            part = pd.DataFrame(
                {
                    "Open": [r[1] for r in rows],
                    "High": [r[1] * 1.0005 for r in rows],
                    "Low": [r[1] * 0.9995 for r in rows],
                    "Close": [r[1] for r in rows],
                    "Volume": 0.0,
                },
                index=pd.DatetimeIndex([r[0] for r in rows]),
            )
            frames.append(part)
        hist = pd.concat(frames)
        plan = {
            "strike": 24000,
            "t_years": 2 / 365.25,
            "iv": 0.15,
            "option_type": "call",
            "stop_premium": 80.0,
            "target_premium": 140.0,
            "entry_premium": 100.0,
            "qty": 65,
            "skip": False,
            "rate": 0.065,
        }
        now = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
        a = monte_carlo(hist, 24000.0, plan, now, n_paths=200, seed=7, costs_cfg=CFG["costs"], exchange="NSE")
        b = monte_carlo(hist, 24000.0, plan, now, n_paths=200, seed=7, costs_cfg=CFG["costs"], exchange="NSE")
        self.assertEqual(a["target_first"], b["target_first"])
        self.assertEqual(a["stop_first"], b["stop_first"])
        self.assertAlmostEqual(a["ev"], b["ev"], places=6)
        self.assertIn(a["label"], ("TAKE", "SKIP"))


class TestReplayNoLookahead(unittest.TestCase):
    def test_future_bars_cannot_affect_past(self):
        closes = [96, 97, 95, 98, 99, 200]
        highs = [100, 99, 98, 101, 102, 210]
        lows = [94, 93, 90, 96, 97, 190]
        df = _bars(closes, highs, lows)
        asof = df.index[4]
        vis = slice_no_lookahead(df, asof.to_pydatetime())
        self.assertEqual(len(vis), 5)
        self.assertTrue((vis.index <= asof).all())
        mutated = df.copy()
        mutated.iloc[-1, mutated.columns.get_loc("Close")] = 1.0
        vis2 = visible_bars(mutated, asof.to_pydatetime())
        sig1 = detect_orb(vis, "BULLISH", CFG, True)
        sig2 = detect_orb(vis2, "BULLISH", CFG, True)
        self.assertEqual(sig1["state"], sig2["state"])
        self.assertEqual(sig1.get("entry"), sig2.get("entry"))

    def test_starting_replay_resets_wall_clock(self):
        app = App(CFG)
        app.last_wall = time.time() - 3600
        prior = _bars([100, 101, 102], start=datetime(2026, 10, 20, 9, 15, tzinfo=IST))
        replay = _bars([103, 104, 105], start=datetime(2026, 10, 21, 9, 15, tzinfo=IST))
        future = _bars([106, 107, 108], start=datetime(2026, 10, 22, 9, 15, tzinfo=IST))
        history = pd.concat([prior, replay, future]).sort_index()
        with patch("trade.load_history_cache", return_value=history):
            app._load_replay(date(2026, 10, 21))
        self.assertGreater(app.last_wall, time.time() - 5)
        ticker = "^NSEI"
        self.assertEqual(app.replay_full.index.date[0], date(2026, 10, 21))
        self.assertTrue((app.history[ticker].index.date < date(2026, 10, 21)).all())


class TestManualExit(unittest.TestCase):
    def test_close_uses_entered_exit_premium(self):
        position = PaperPosition(
            index_name="NIFTY",
            setup="ORB",
            direction="long",
            contract="NIFTY 24500 CE",
            option_type="call",
            strike=24500,
            expiry=date(2026, 10, 27),
            iv=0.15,
            rate=0.065,
            lot=65,
            lots=1,
            qty=65,
            remaining_qty=65,
            entry_time=datetime(2026, 10, 21, 10, 0, tzinfo=IST),
            entry_index=24500,
            entry_premium=100,
            stop_index=24400,
            target_index=24600,
            partial_index=24550,
            stop_premium=70,
            target_premium=140,
            exchange="NSE",
            costs_cfg=CFG["costs"],
            time_stop_minutes=40,
        )
        exit_time = datetime(2026, 10, 21, 10, 10, tzinfo=IST)
        with patch("core.journal.append_row") as append_row:
            result = position.close(24500, exit_time, "manual_exit", exit_premium=123.45)
        self.assertAlmostEqual(result["exit_premium"], 123.45)
        self.assertAlmostEqual(float(append_row.call_args.args[0]["exit_premium"]), 123.45)


if __name__ == "__main__":
    unittest.main()
