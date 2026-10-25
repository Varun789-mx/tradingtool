from __future__ import annotations

import logging
import time
import warnings
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*yfinance.*")
warnings.filterwarnings("ignore", message=".*pandas.*")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

IST = ZoneInfo("Asia/Kolkata")
LOG = logging.getLogger("trade")

INDEX_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}
VIX_TICKER = "^INDIAVIX"
GLOBAL_TICKERS = [
    "ES=F",
    "NQ=F",
    "^GSPC",
    "^IXIC",
    "^N225",
    "^HSI",
    "CL=F",
    "GC=F",
    "USDINR=X",
    "DX-Y.NYB",
]

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def ist_now() -> datetime:
    return datetime.now(IST)


def to_ist_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        di = di.tz_localize("UTC")
    return di.tz_convert(IST)


def parse_holidays(cfg: dict[str, Any]) -> set[date]:
    out: set[date] = set()
    for item in cfg.get("holidays", []):
        out.add(date.fromisoformat(str(item)[:10]))
    return out


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def is_trading_day(d: date, holidays: set[date]) -> bool:
    return not is_weekend(d) and d not in holidays


def previous_trading_day(d: date, holidays: set[date]) -> date:
    cur = d
    while not is_trading_day(cur, holidays):
        cur -= timedelta(days=1)
    return cur


def next_trading_day(d: date, holidays: set[date]) -> date:
    cur = d
    while not is_trading_day(cur, holidays):
        cur += timedelta(days=1)
    return cur


def session_phase(now: datetime, holidays: set[date]) -> str:
    now = now.astimezone(IST)
    d = now.date()
    t = now.time()
    if not is_trading_day(d, holidays):
        return "CLOSED"
    if t < dtime(8, 0):
        return "CLOSED"
    if t < dtime(9, 15):
        return "PRE_MARKET"
    if t < dtime(9, 30):
        return "OPENING_RANGE"
    if t < dtime(14, 30):
        return "ACTIVE"
    if t < dtime(15, 15):
        return "MANAGE"
    if t < dtime(15, 30):
        return "SQUARE_OFF"
    return "POST_MARKET"


def next_open(now: datetime, holidays: set[date]) -> datetime:
    now = now.astimezone(IST)
    d = now.date()
    open_t = datetime.combine(d, dtime(9, 15), tzinfo=IST)
    if is_trading_day(d, holidays) and now < open_t:
        return open_t
    nxt = next_trading_day(d + timedelta(days=1), holidays)
    return datetime.combine(nxt, dtime(9, 15), tzinfo=IST)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]) for c in out.columns]
    out.columns = [str(c) for c in out.columns]
    return out


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    lower = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    out = pd.DataFrame(index=to_ist_index(df.index))
    for std in ("Open", "High", "Low", "Close"):
        key = std.lower()
        if key not in lower:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        out[std] = pd.to_numeric(df[lower[key]].to_numpy(), errors="coerce")
    if "volume" in lower:
        out["Volume"] = pd.to_numeric(df[lower["volume"]].to_numpy(), errors="coerce")
    else:
        out["Volume"] = 0.0
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _frames_from_yf(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {t: pd.DataFrame() for t in tickers}
    if not isinstance(raw.columns, pd.MultiIndex):
        t0 = tickers[0]
        return {t0: _normalize_ohlc(raw)}
    level0 = set(map(str, raw.columns.get_level_values(0)))
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        part = None
        if t in level0:
            part = raw[t]
        else:
            try:
                part = raw.xs(t, axis=1, level=1)
            except (KeyError, ValueError):
                part = None
        result[t] = _normalize_ohlc(part) if part is not None else pd.DataFrame()
    return result


def fetch_ohlc(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return _frames_from_yf(raw, tickers)


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = to_ist_index(out.index)
    out["typical"] = (out["High"] + out["Low"] + out["Close"]) / 3.0
    parts: list[pd.DataFrame] = []
    dates = pd.Series(out.index.date, index=out.index)
    for _, g in out.groupby(dates):
        g = g.copy()
        g["session_avg"] = g["typical"].expanding().mean()
        parts.append(g)
    return pd.concat(parts).sort_index() if parts else out


def session_slice(df: pd.DataFrame, day: date | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = to_ist_index(out.index)
    if day is not None:
        out = out[out.index.date == day]
    times = out.index.tz_convert(IST)
    mask = (times.time >= dtime(9, 15)) & (times.time < dtime(15, 30))
    return out.loc[mask]


def closed_and_forming(
    df: pd.DataFrame, now: datetime, bar_minutes: int = 5
) -> tuple[pd.DataFrame, pd.Series | None]:
    if df is None or df.empty:
        return pd.DataFrame(), None
    now = now.astimezone(IST)
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(IST)
    else:
        last_ts = last_ts.tz_convert(IST)
    bar_end = last_ts + timedelta(minutes=bar_minutes)
    last = df.iloc[-1]
    if now < bar_end:
        return df.iloc[:-1].copy(), last
    return df.copy(), last


def data_age_seconds(df: pd.DataFrame, now: datetime) -> float | None:
    if df is None or df.empty:
        return None
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize(IST)
    return max(0.0, (now.astimezone(IST) - last.tz_convert(IST)).total_seconds())


def is_stale(df: pd.DataFrame, now: datetime, holidays: set[date], limit_sec: int = 600) -> bool:
    phase = session_phase(now, holidays)
    if phase in ("CLOSED", "PRE_MARKET", "POST_MARKET"):
        return False
    age = data_age_seconds(df, now)
    if age is None:
        return True
    return age > limit_sec


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = add_session_features(df)
    close = out["Close"]
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    prev = close.shift(1)
    tr = pd.concat(
        [
            (out["High"] - out["Low"]).abs(),
            (out["High"] - prev).abs(),
            (out["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    return out


def history_cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("^", "").replace("=", "").replace("-", "")
    return CACHE_DIR / f"hist_{safe}.pkl"


def cache_is_fresh(path: Path, now: datetime) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
    return mtime.astimezone(IST).date() == now.astimezone(IST).date()


def load_history_cache(ticker: str) -> pd.DataFrame:
    path = history_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_pickle(path)
        return with_indicators(_normalize_ohlc(df)) if not df.empty else pd.DataFrame()
    except Exception:
        LOG.exception("history cache read failed %s", ticker)
        return pd.DataFrame()


def save_history_cache(ticker: str, df: pd.DataFrame) -> None:
    path = history_cache_path(ticker)
    try:
        df.to_pickle(path)
    except Exception:
        LOG.exception("history cache write failed %s", ticker)


def fetch_history_59d(ticker: str) -> pd.DataFrame:
    frames = fetch_ohlc([ticker], period="59d", interval="5m")
    df = frames.get(ticker, pd.DataFrame())
    return with_indicators(df)


def global_changes(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, df in frames.items():
        if df is None or df.empty or len(df) < 1:
            rows.append({"ticker": t, "last": None, "chg_pct": None})
            continue
        last = float(df["Close"].iloc[-1])
        chg = None
        if len(df) >= 2:
            prev = float(df["Close"].iloc[-2])
            if prev:
                chg = (last - prev) / prev * 100.0
        rows.append({"ticker": t, "last": last, "chg_pct": chg})
    return rows


def fetch_with_backoff(fn, last_good, backoff_state: dict[str, float], key: str):
    """Call fn(); on failure keep last_good and grow backoff_state[key]."""
    try:
        result = fn()
        backoff_state[key] = 0.0
        return result, True, None
    except Exception as exc:
        LOG.exception("fetch failed %s", key)
        backoff_state[key] = min(300.0, max(30.0, float(backoff_state.get(key, 0.0) or 15.0) * 2.0))
        return last_good, False, str(exc)


def sleep_backoff_note(backoff_state: dict[str, float], key: str) -> float:
    return float(backoff_state.get(key, 0.0) or 0.0)


def gap_vs_prev_close(session: pd.DataFrame, history: pd.DataFrame) -> float | None:
    if session is None or session.empty or history is None or history.empty:
        return None
    day = session.index[-1].date()
    prev = history[history.index.date < day]
    if prev.empty or session.empty:
        return None
    prev_close = float(prev["Close"].iloc[-1])
    open_px = float(session["Open"].iloc[0])
    if not prev_close:
        return None
    return (open_px - prev_close) / prev_close * 100.0


def expected_gap(globals_rows: list[dict[str, Any]]) -> str:
    """Heuristic pre-open gap cue from US futures / Asia — not a forecast."""
    by = {r["ticker"]: r.get("chg_pct") for r in globals_rows}
    bits = []
    es = by.get("ES=F")
    nq = by.get("NQ=F")
    nk = by.get("^N225")
    hs = by.get("^HSI")
    if es is not None:
        bits.append(f"ES {es:+.2f}%")
    if nq is not None:
        bits.append(f"NQ {nq:+.2f}%")
    if nk is not None:
        bits.append(f"NK {nk:+.2f}%")
    if hs is not None:
        bits.append(f"HSI {hs:+.2f}%")
    avg_vals = [v for v in (es, nq, nk, hs) if v is not None]
    if not avg_vals:
        return "expected gap: n/a (" + ", ".join(bits) + ")" if bits else "expected gap: n/a"
    avg = sum(avg_vals) / len(avg_vals)
    tone = "soft bid" if avg > 0.15 else "soft offer" if avg < -0.15 else "mixed"
    return f"expected gap cue: {tone} ({', '.join(bits)}) — not a prediction"


def retry_download(tickers: list[str], period: str, interval: str, attempts: int = 1) -> dict[str, pd.DataFrame]:
    last: dict[str, pd.DataFrame] = {}
    err = None
    for i in range(max(1, attempts)):
        try:
            return fetch_ohlc(tickers, period, interval)
        except Exception as exc:
            err = exc
            time.sleep(min(2**i, 8))
    LOG.warning("download exhausted %s %s", tickers, err)
    return last
