from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
LEXICON_PATH = ROOT / "data" / "sentiment_lexicon.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "capital": 100000.0,
    "max_loss_pct_per_trade": 1.0,
    "daily_loss_limit": 2000.0,
    "daily_profit_goal": 3000.0,
    "max_trades_per_day": 2,
    "r_multiple_target": 1.5,
    "partial_at_R": 1.0,
    "time_stop_minutes": 40,
    "entry_window": ["09:30", "14:30"],
    "hard_exit": "15:15",
    "auto_journal_take": True,
    "mc_paths": 10000,
    "mc_seed": 42,
    "risk_free": 0.065,
    "_holidays_note": "Keep this list updated. If an expiry falls on a holiday, it shifts to the previous trading day.",
    "holidays": ["2026-10-20", "2026-11-10", "2026-11-24"],
    "selected_index": "NIFTY",
    "indices": {
        "NIFTY": {
            "ticker": "^NSEI",
            "lot": 65,
            "step": 50,
            "expiry": "weekly_tuesday",
            "exchange": "NSE",
            "iv_scalar": 1.0,
        },
        "BANKNIFTY": {
            "ticker": "^NSEBANK",
            "lot": 30,
            "step": 100,
            "expiry": "monthly_last_tuesday",
            "exchange": "NSE",
            "iv_scalar": 1.0,
        },
        "SENSEX": {
            "ticker": "^BSESN",
            "lot": 20,
            "step": 100,
            "expiry": "weekly_thursday",
            "exchange": "BSE",
            "iv_scalar": 1.0,
        },
    },
    "costs": {
        "brokerage_per_order": 20.0,
        "stt_sell_premium_pct": 0.1,
        "stt_changed_recently": True,
        "exchange_txn_nse_pct": 0.03503,
        "exchange_txn_bse_pct": 0.0325,
        "sebi_fee_pct": 0.0001,
        "stamp_duty_buy_pct": 0.003,
        "gst_pct": 18.0,
        "slippage_per_unit": 1.0,
    },
    "orb": {"min_range_pct": 0.15, "max_range_pct": 0.80},
    "bias_weights": {
        "us_futures": 15,
        "asia": 10,
        "crude": 8,
        "usdinr": 8,
        "dxy": 5,
        "vix_level": 10,
        "vix_change": 8,
        "gap": 12,
        "news": 12,
        "intraday_trend": 12,
    },
    "events": [
        {"date": "2026-10-28", "name": "US CPI", "kind": "US_CPI"},
        {"date": "2026-10-29", "name": "FOMC", "kind": "US_FED"},
        {"date": "2026-12-05", "name": "RBI Policy", "kind": "RBI"},
    ],
    "news_queries": [
        "Nifty",
        "Sensex",
        "Bank Nifty",
        "RBI",
        "Fed",
        "FII",
        "crude oil",
    ],
}

DEFAULT_LEXICON: dict[str, Any] = {
    "positive": {
        "rally": 2,
        "surge": 2,
        "rate cut": 3,
        "cut rates": 3,
        "fii buying": 2,
        "record high": 2,
        "upgrade": 1,
        "inflow": 1,
        "peace": 1,
        "strong": 1,
        "bull": 1,
        "growth": 1,
        "rebound": 1,
        "optimistic": 1,
    },
    "negative": {
        "crash": 4,
        "war": 4,
        "rate hike": 3,
        "hike": 2,
        "fii selling": 2,
        "default": 4,
        "ban": 3,
        "circuit": 4,
        "selloff": 2,
        "sell-off": 2,
        "recession": 3,
        "geopolit": 2,
        "outflow": 2,
        "weak": 1,
        "bear": 1,
        "slump": 2,
        "tumble": 2,
        "tension": 2,
        "inflation shock": 2,
    },
    "market_moving": [
        "rate cut",
        "rate hike",
        "war",
        "crash",
        "ban",
        "default",
        "circuit",
    ],
    "tags": {
        "RBI": ["rbi", "repo", "mpc"],
        "Fed": ["fed", "fomc", "powell", "federal reserve"],
        "results": ["results", "earnings", "profit"],
        "geopolitics": ["war", "geopolit", "missile", "conflict"],
        "FII/DII": ["fii", "dii", "foreign fund"],
        "crude": ["crude", "oil", "brent", "wti"],
        "budget": ["budget", "fiscal"],
    },
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_config(cfg: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, raw)


def ensure_lexicon(path: Path = LEXICON_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_LEXICON, indent=2), encoding="utf-8")
        return dict(DEFAULT_LEXICON)
    try:
        return deep_merge(DEFAULT_LEXICON, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_LEXICON)


def first_run_prompt(path: Path = CONFIG_PATH) -> dict[str, Any]:
    print("First run — decision support only; no orders will ever be placed.")
    print("WARNING: NSE lot sizes may have been revised (circular dated 3 Oct 2026). Verify on NSE/BSE before sizing.")
    raw_cap = input("Capital (INR): ").strip() or "100000"
    raw_loss = input("Max loss % per trade [1]: ").strip() or "1"
    cfg = dict(DEFAULT_CONFIG)
    cfg["capital"] = float(raw_cap)
    cfg["max_loss_pct_per_trade"] = float(raw_loss)
    save_config(cfg, path)
    ensure_lexicon()
    return cfg


def ensure_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return first_run_prompt(path)
    cfg = load_config(path)
    ensure_lexicon()
    return cfg
