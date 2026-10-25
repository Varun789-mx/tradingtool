from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
import webbrowser
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*yfinance.*")
warnings.filterwarnings("ignore", message=".*pandas.*")

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.bias import compute_bias, what_changed
from core.chart import render_live_chart, save_trade_chart
from core.config import ensure_config
from core.data import (
    GLOBAL_TICKERS,
    INDEX_TICKERS,
    IST,
    VIX_TICKER,
    cache_is_fresh,
    closed_and_forming,
    data_age_seconds,
    fetch_history_59d,
    fetch_ohlc,
    gap_vs_prev_close,
    global_changes,
    history_cache_path,
    is_stale,
    ist_now,
    load_history_cache,
    next_open,
    parse_holidays,
    save_history_cache,
    session_phase,
    session_slice,
    with_indicators,
    expected_gap,
)
from core.journal import day_stats, PaperPosition, position_from_plan
from core.news import collect_news, mark_new, market_moving, news_score
from core.options import build_plan, implied_vol, years_to_expiry
from core.signals import evaluate_setups
from core.timemachine import (
    history_setup_stats,
    monte_carlo,
    slice_no_lookahead,
    advance_replay,
    replay_clock_start,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("trade")
CONSOLE = Console()
SEBI_NOTE = "Most option buyers lose money. This is decision support, not advice, not a forecast, not an order router."


def beep() -> None:
    try:
        import winsound

        winsound.Beep(880, 350)
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


def poll_key() -> str | None:
    try:
        if sys.platform == "win32":
            import msvcrt

            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0") and msvcrt.kbhit():
                    msvcrt.getwch()
                    return None
                return ch
        else:
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
    except Exception:
        return None
    return None


class App:
    def __init__(self, cfg: dict[str, Any], replay_day: date | None = None, speed: float = 20.0):
        self.cfg = cfg
        self.replay_day = replay_day
        self.speed = speed
        self.selected = cfg.get("selected_index") or "NIFTY"
        self.holidays = parse_holidays(cfg)
        self.frames: dict[str, Any] = {}
        self.vix = None
        self.globals_rows: list[dict[str, Any]] = []
        self.news: list[dict[str, Any]] = []
        self.seen_news: set[str] = set()
        self.bias: dict[str, Any] | None = None
        self.prev_bias: dict[str, Any] | None = None
        self.changed = ""
        self.setups: list[dict[str, Any]] = []
        self.plan: dict[str, Any] | None = None
        self.tm: dict[str, Any] | None = None
        self.hist_stats: dict[str, Any] | None = None
        self.paper: PaperPosition | None = None
        self.events: list[str] = []
        self.stale = False
        self.fetch_ok = True
        self.live_premium: float | None = None
        self.iv_override: float | None = None
        self.input_mode: str | None = None
        self.buffer = ""
        self.show_help = False
        self.chart_path = None
        self.premium_trace: list = []
        self.pnl_trace: list = []
        self.day_pnl_curve: list = []
        self.fired: set[str] = set()
        self.trade_chart_n = 0
        self.backoff: dict[str, float] = {}
        self.last_run: dict[str, float] = {"index": 0.0, "global": 0.0, "news": 0.0, "history": 0.0}
        self.history: dict[str, Any] = {}
        self.sim_now: datetime | None = None
        self.replay_full = None
        self.replay_vix = None
        self.wall0 = time.time()
        self.last_wall = time.time()
        if replay_day:
            self._load_replay(replay_day)

    def now(self) -> datetime:
        if self.sim_now is not None:
            return self.sim_now
        return ist_now()

    def log_event(self, msg: str) -> None:
        ts = self.now().strftime("%H:%M:%S")
        self.events.append(f"{ts} {msg}")
        self.events = self.events[-10:]
        LOG.info(msg)

    def _load_replay(self, day: date) -> None:
        ticker = INDEX_TICKERS[self.selected]
        hist = load_history_cache(ticker)
        if hist is None or hist.empty:
            try:
                hist = fetch_history_59d(ticker)
                save_history_cache(ticker, hist)
            except Exception:
                LOG.exception("replay history fetch")
                hist = load_history_cache(ticker)
        self.replay_full = session_slice(hist, day)
        self.history[ticker] = hist[hist.index.date < day].copy() if hist is not None and not hist.empty else hist
        vix = load_history_cache(VIX_TICKER)
        if vix is None or vix.empty:
            try:
                vix = fetch_history_59d(VIX_TICKER)
                save_history_cache(VIX_TICKER, vix)
            except Exception:
                vix = load_history_cache(VIX_TICKER)
        self.replay_vix = session_slice(vix, day) if vix is not None else None
        self.history[VIX_TICKER] = (
            vix[vix.index.date < day].copy() if vix is not None and not vix.empty else vix
        )
        self.sim_now = replay_clock_start(day)
        self.last_wall = time.time()
        self.log_event(f"REPLAY {day.isoformat()} news disabled — no lookahead")

    def due(self, name: str, every: float) -> bool:
        now_m = time.time()
        extra = self.backoff.get(name, 0.0)
        if now_m - self.last_run.get(name, 0.0) >= every + extra:
            return True
        return False

    def mark_ran(self, name: str, ok: bool) -> None:
        self.last_run[name] = time.time()
        if ok:
            self.backoff[name] = 0.0
        else:
            self.backoff[name] = min(300.0, max(30.0, (self.backoff.get(name, 0.0) or 15.0) * 2.0))

    def refresh_index(self) -> None:
        if self.replay_day is not None:
            vis = slice_no_lookahead(self.replay_full, self.now()) if self.replay_full is not None else None
            self.frames[INDEX_TICKERS[self.selected]] = with_indicators(vis) if vis is not None else vis
            if self.replay_vix is not None and not self.replay_vix.empty:
                self.vix = slice_no_lookahead(self.replay_vix, self.now())
            self.fetch_ok = True
            self.mark_ran("index", True)
            return
        tickers = list(INDEX_TICKERS.values()) + [VIX_TICKER]
        try:
            frames = fetch_ohlc(tickers, period="5d", interval="5m")
            for t, df in frames.items():
                if df is not None and not df.empty:
                    self.frames[t] = with_indicators(df)
            self.vix = self.frames.get(VIX_TICKER)
            self.fetch_ok = True
            self.mark_ran("index", True)
        except Exception:
            LOG.exception("index fetch")
            self.fetch_ok = False
            self.mark_ran("index", False)

    def refresh_globals(self) -> None:
        if self.replay_day is not None:
            self.mark_ran("global", True)
            return
        try:
            frames = fetch_ohlc(GLOBAL_TICKERS, period="5d", interval="1d")
            self.globals_rows = global_changes(frames)
            self.mark_ran("global", True)
        except Exception:
            LOG.exception("global fetch")
            self.mark_ran("global", False)

    def refresh_news(self) -> None:
        if self.replay_day is not None:
            self.news = []
            self.mark_ran("news", True)
            return
        try:
            items = collect_news(self.cfg, self.now())
            items, seen = mark_new(items, self.seen_news)
            moving = market_moving(items)
            self.seen_news = seen
            self.news = items
            for m in moving:
                self.log_event(f"MARKET-MOVING: {m['title'][:80]}")
                beep()
            self.mark_ran("news", True)
        except Exception:
            LOG.exception("news fetch")
            self.mark_ran("news", False)

    def refresh_history(self) -> None:
        for t in list(INDEX_TICKERS.values()) + [VIX_TICKER]:
            path_ok = cache_is_fresh(history_cache_path(t), self.now())
            if path_ok and t in self.history and self.history[t] is not None and not getattr(self.history[t], "empty", True):
                continue
            cached = load_history_cache(t)
            if cached is not None and not cached.empty and path_ok:
                self.history[t] = self._history_before_replay(cached)
                continue
            if self.replay_day is not None and cached is not None and not cached.empty:
                self.history[t] = self._history_before_replay(cached)
                continue
            try:
                df = fetch_history_59d(t)
                save_history_cache(t, df)
                self.history[t] = self._history_before_replay(df)
            except Exception:
                LOG.exception("history %s", t)
                if cached is not None and not cached.empty:
                    self.history[t] = self._history_before_replay(cached)
        self.mark_ran("history", True)
        try:
            ticker = INDEX_TICKERS[self.selected]
            self.hist_stats = history_setup_stats(
                self.history.get(ticker),
                self.cfg,
                self.selected,
                self.now(),
                use_cache=self.replay_day is None,
            )
        except Exception:
            LOG.exception("history stats")

    def _history_before_replay(self, history):
        if self.replay_day is None or history is None or history.empty:
            return history
        return history[history.index.date < self.replay_day].copy()

    def session_df(self):
        ticker = INDEX_TICKERS[self.selected]
        df = self.frames.get(ticker)
        if df is None or getattr(df, "empty", True):
            hist = self.history.get(ticker)
            if hist is not None:
                df = hist
        if df is None:
            return None
        return session_slice(df, self.now().date())

    def recompute(self) -> None:
        now = self.now()
        phase = session_phase(now, self.holidays)
        sess = self.session_df()
        closed, forming = closed_and_forming(sess, now) if sess is not None else (None, None)
        self.stale = is_stale(sess, now, self.holidays) if sess is not None else True
        if sess is None or sess.empty:
            self.stale = phase in ("OPENING_RANGE", "ACTIVE", "MANAGE", "SQUARE_OFF")
        gap = gap_vs_prev_close(sess, self.history.get(INDEX_TICKERS[self.selected])) if sess is not None else None
        nscore = news_score(self.news)
        stats = day_stats(now.date())
        halt = self._halt(stats)
        self.prev_bias = self.bias
        self.bias = compute_bias(
            self.cfg,
            self.globals_rows,
            self.vix if self.vix is not None else self.history.get(VIX_TICKER),
            sess if sess is not None else None,
            gap,
            nscore,
            now,
            self.holidays,
            self.stale,
            halt,
        )
        self.changed = what_changed(self.prev_bias, self.bias)
        allow = phase == "ACTIVE" and not self.stale and not halt
        extra = None
        if self.stale and phase in ("ACTIVE", "MANAGE", "OPENING_RANGE"):
            extra = "STALE"
        self.setups = evaluate_setups(
            closed if closed is not None else None,
            self.bias["label"],
            self.cfg,
            allow,
            self.stale,
            extra,
        )
        self.plan = None
        self.tm = None
        triggered = [s for s in self.setups if s.get("state") == "TRIGGERED"]
        spot = None
        if sess is not None and not sess.empty:
            spot = float((forming["Close"] if forming is not None else sess["Close"].iloc[-1]))
        if triggered and spot is not None:
            sig = triggered[0]
            iv = self._iv()
            try:
                self.plan = build_plan(
                    self.cfg,
                    self.selected,
                    spot,
                    sig["direction"],
                    float(sig["entry"]),
                    float(sig["stop"]),
                    float(sig["target"]),
                    now,
                    iv,
                    self.live_premium,
                )
            except Exception:
                LOG.exception("plan")
                self.plan = None
            if self.plan and not self.plan.get("skip"):
                pos_ok = True
                if self.hist_stats:
                    st = (self.hist_stats.get("setups") or {}).get(sig["name"])
                    if st and not st.get("positive_expectancy", True):
                        pos_ok = False
                        self.plan["hist_skip"] = True
                try:
                    self.tm = monte_carlo(
                        self.history.get(INDEX_TICKERS[self.selected]),
                        spot,
                        {**self.plan, "rate": self.cfg["risk_free"]},
                        now,
                        n_paths=int(self.cfg.get("mc_paths", 10000)),
                        seed=int(self.cfg.get("mc_seed", 42)),
                        costs_cfg=self.cfg["costs"],
                        exchange=self.plan["exchange"],
                    )
                except Exception:
                    LOG.exception("monte carlo")
                if self.tm and (self.tm.get("label") != "TAKE" or not pos_ok or halt):
                    if self.tm.get("label") == "TAKE" and not pos_ok:
                        self.tm["label"] = "SKIP"
                        self.tm["verdict"] = "SKIP: historical net expectancy not positive for this setup."
                if (
                    self.tm
                    and self.tm.get("label") == "TAKE"
                    and allow
                    and self.cfg.get("auto_journal_take")
                    and self.paper is None
                    and sig["name"] not in self.fired
                    and not halt
                ):
                    self.paper = position_from_plan(self.plan, sig, now, self.cfg)
                    self.fired.add(sig["name"])
                    self.log_event(f"PAPER TAKE {self.plan['contract']} {sig['name']}")
                    beep()
        self._manage_paper(forming if forming is not None else (sess.iloc[-1] if sess is not None and not sess.empty else None), now, phase)

    def _iv(self) -> float:
        if self.iv_override:
            return float(self.iv_override)
        scalar = float(self.cfg["indices"][self.selected].get("iv_scalar", 1.0))
        vix = None
        if self.vix is not None and not self.vix.empty:
            vix = float(self.vix["Close"].iloc[-1])
        elif self.history.get(VIX_TICKER) is not None and not self.history[VIX_TICKER].empty:
            vix = float(self.history[VIX_TICKER]["Close"].iloc[-1])
        if vix is None:
            return 0.15 * scalar
        return max(0.05, (vix / 100.0) * scalar)

    def _halt(self, stats: dict[str, Any]) -> bool:
        if stats["pnl"] <= -abs(float(self.cfg["daily_loss_limit"])):
            return True
        if stats["pnl"] >= float(self.cfg["daily_profit_goal"]) and stats["trades"] > 0:
            return True
        open_n = 1 if self.paper else 0
        if stats["trades"] + open_n >= int(self.cfg["max_trades_per_day"]):
            return True
        return False

    def _manage_paper(self, bar, now: datetime, phase: str) -> None:
        if self.paper is None or bar is None:
            return
        hard = phase == "SQUARE_OFF" or now.time() >= dtime(15, 15)
        try:
            mark = self.paper.mark_premium(float(bar["Close"]), now)
            self.premium_trace.append((now, mark))
            self.premium_trace = self.premium_trace[-200:]
            _, _, net = self.paper.net_if_exit(float(bar["Close"]), now)
            self.pnl_trace.append((now, net))
            self.pnl_trace = self.pnl_trace[-200:]
        except Exception:
            LOG.exception("mark")
        try:
            result = self.paper.manage(bar, now, hard)
        except Exception:
            LOG.exception("manage")
            result = None
        if self.paper and self.paper.events:
            for e in self.paper.events:
                if "partial" in e:
                    self.log_event("1R partial — stop to cost")
                    beep()
            self.paper.events.clear()
        if result:
            self.log_event(f"PAPER EXIT {result['reason']} net {result['net']:.0f}")
            beep()
            self.trade_chart_n += 1
            try:
                save_trade_chart(
                    self.selected,
                    self.session_df(),
                    self.plan,
                    self.paper,
                    self.premium_trace,
                    self.pnl_trace,
                    now.date(),
                    self.trade_chart_n,
                )
            except Exception:
                LOG.exception("trade chart")
            self.paper = None
        if phase == "SQUARE_OFF" and self.paper:
            self.log_event("SQUARE OFF — exit everything")

    def handle_key(self, ch: str) -> bool:
        if self.input_mode:
            if ch in ("\r", "\n"):
                self._commit_input()
                return True
            if ch in ("\x08", "\x7f"):
                self.buffer = self.buffer[:-1]
                return True
            if ch == "\x1b":
                self.input_mode = None
                self.buffer = ""
                return True
            if ch.isprintable():
                self.buffer += ch
            return True
        k = ch.lower() if ch else ""
        if k == "q":
            return False
        if k == "n":
            self.selected = "NIFTY"
            self.log_event("index NIFTY")
        elif k == "b":
            self.selected = "BANKNIFTY"
            self.log_event("index BANKNIFTY")
        elif k == "s":
            self.selected = "SENSEX"
            self.log_event("index SENSEX")
        elif k == "h":
            self.show_help = not self.show_help
        elif k == "t":
            self.recompute()
            self.log_event("Time Machine recomputed")
        elif k == "c":
            try:
                path = render_live_chart(
                    self.selected,
                    self.session_df(),
                    self.plan,
                    self.paper,
                    self.premium_trace,
                    self.pnl_trace,
                )
                webbrowser.open(path.resolve().as_uri())
                self.log_event("opened chart")
            except Exception:
                LOG.exception("chart")
                self.log_event("chart failed")
        elif k == "p":
            self.input_mode = "premium"
            self.buffer = ""
        elif k == "e":
            self.input_mode = "entry"
            self.buffer = ""
        elif k == "x":
            self.input_mode = "exit"
            self.buffer = ""
        elif k == "r":
            self.input_mode = "replay"
            self.buffer = ""
        return True

    def _commit_input(self) -> None:
        mode, buf = self.input_mode, self.buffer.strip()
        self.input_mode = None
        self.buffer = ""
        if not buf:
            return
        if mode == "replay":
            try:
                day = date.fromisoformat(buf)
                self.replay_day = day
                self._load_replay(day)
            except ValueError:
                self.log_event("replay date YYYY-MM-DD")
            return
        try:
            px = float(buf)
        except ValueError:
            self.log_event("need a number")
            return
        if px <= 0:
            self.log_event("premium must be positive")
            return
        if mode == "premium" and self.plan:
            try:
                t = years_to_expiry(self.now(), self.plan["expiry"])
                self.iv_override = implied_vol(
                    px,
                    float(self.plan["spot"]),
                    float(self.plan["strike"]),
                    max(t, 1e-8),
                    float(self.cfg["risk_free"]),
                    self.plan["option_type"],
                )
                self.live_premium = px
                self.log_event(f"IV calibrated {self.iv_override:.3f} from premium {px}")
            except Exception:
                LOG.exception("iv")
                self.log_event("IV solver failed")
        elif mode == "entry":
            self.live_premium = px
            if self.plan and self.paper is None and not self.plan.get("skip"):
                plan = dict(self.plan)
                plan["entry_premium"] = px
                plan["live_premium"] = px
                setup = next((s for s in self.setups if s.get("state") == "TRIGGERED"), {"name": "MANUAL", "partial": plan.get("entry_index")})
                self.paper = position_from_plan(plan, setup, self.now(), self.cfg)
                self.log_event(f"I ENTERED {plan['contract']} @ {px}")
            else:
                self.log_event(f"live premium stored {px}")
        elif mode == "exit":
            if self.paper:
                spot = float(self.session_df()["Close"].iloc[-1])
                res = self.paper.close(spot, self.now(), "manual_exit", exit_premium=px)
                self.log_event(f"I EXITED net {res['net']:.0f}")
                self.paper = None
            else:
                self.log_event("no paper position")

    def tick_tasks(self) -> None:
        now = self.now()
        phase = session_phase(now, self.holidays)
        if self.replay_day is not None:
            wall = time.time()
            self.sim_now = advance_replay(self.sim_now, self.speed, wall - self.last_wall)
            self.last_wall = wall
            end = datetime.combine(self.replay_day, dtime(16, 0), tzinfo=IST)
            if self.sim_now > end:
                self.sim_now = end
        mkt = phase in ("OPENING_RANGE", "ACTIVE", "MANAGE", "SQUARE_OFF", "PRE_MARKET")
        idx_every = 30.0 if mkt and not self.replay_day else (1.0 if self.replay_day else 120.0)
        if self.due("index", idx_every):
            self.refresh_index()
            self.recompute()
            try:
                render_live_chart(self.selected, self.session_df(), self.plan, self.paper, self.premium_trace, self.pnl_trace)
            except Exception:
                LOG.exception("chart rewrite")
        if self.due("global", 120.0):
            self.refresh_globals()
            self.recompute()
        if self.due("news", 300.0):
            self.refresh_news()
            self.recompute()
        if self.due("history", 3600.0):
            self.refresh_history()

    def render(self) -> Layout:
        now = self.now()
        phase = session_phase(now, self.holidays)
        sess = self.session_df()
        spot = float(sess["Close"].iloc[-1]) if sess is not None and not sess.empty else None
        age = data_age_seconds(sess, now) if sess is not None else None
        vix = None
        if self.vix is not None and not self.vix.empty:
            vix = float(self.vix["Close"].iloc[-1])
        header = Text()
        header.append(now.strftime("%Y-%m-%d %H:%M:%S IST  "), style="bold")
        header.append(f"{phase}  ", style="bold yellow" if phase == "SQUARE_OFF" else "bold cyan")
        header.append(f"{self.selected} {spot:.0f}  " if spot else f"{self.selected} n/a  ")
        header.append(f"VIX {vix:.2f}  " if vix else "VIX n/a  ")
        if age is not None:
            header.append(f"age {int(age)}s  ", style="red" if self.stale else "green")
        if self.stale:
            header.append(" STALE ", style="bold white on red")
        if not self.fetch_ok:
            header.append(" data stale ", style="black on yellow")
        header.append(" PAPER / DECISION-SUPPORT ONLY ", style="bold white on blue")
        if self.replay_day:
            header.append(f" REPLAY {self.replay_day} {self.speed}x ", style="black on magenta")

        layout = Layout()
        layout.split_column(
            Layout(Panel(header, title="trade.py"), name="head", size=3),
            Layout(name="mid"),
            Layout(name="low"),
            Layout(name="foot", size=3),
        )
        layout["mid"].split_row(Layout(name="left"), Layout(name="center"), Layout(name="right"))
        layout["low"].split_row(Layout(name="tm"), Layout(name="pos"), Layout(name="stats"))

        layout["left"].update(self._bias_news_panel(phase))
        layout["center"].update(self._setups_plan())
        layout["right"].update(self._globals_events())
        layout["tm"].update(self._tm_panel())
        layout["pos"].update(self._paper_panel(now, phase, spot))
        layout["stats"].update(self._stats_panel(now, phase))
        keys = "n Nifty  b Bank  s Sensex  t TimeMachine  c chart  p premium  e I entered  x exit  r replay  h help  q quit"
        if self.input_mode:
            keys = f"{self.input_mode}> {self.buffer}_   (Enter commit, Esc cancel)"
        layout["foot"].update(Panel(Text(keys + "\n" + SEBI_NOTE, style="dim"), title="keys"))
        if self.show_help:
            layout["center"].update(Panel(_help_text(), title="help"))
        return layout

    def _bias_news_panel(self, phase: str) -> Panel:
        b = self.bias or {}
        t = Table.grid(expand=True)
        score = b.get("score", 0)
        label = b.get("label", "n/a")
        arrow = "→"
        if "▲" in self.changed:
            arrow = "▲"
        elif "▼" in self.changed:
            arrow = "▼"
        style = "green" if label == "BULLISH" else "red" if label == "BEARISH" else "yellow"
        t.add_row(Text(f"{label} {score:+.0f} {arrow}", style=f"bold {style}"))
        t.add_row(Text(self.changed or "", style="dim"))
        t.add_row(Text(b.get("news_note") or "News score is a bias input, never a forecast.", style="dim italic"))
        if phase == "PRE_MARKET":
            t.add_row(Text(expected_gap(self.globals_rows), style="cyan"))
        for f in (b.get("factors") or [])[:10]:
            mark = "PASS" if f["ok"] else "FAIL"
            col = "green" if f["ok"] else "red"
            t.add_row(Text(f"{mark:4} {f['name']}: {f['detail']}", style=col))
        news_tbl = Table(title="News (latest 8)", expand=True, box=None)
        news_tbl.add_column("h", ratio=1)
        if self.replay_day:
            news_tbl.add_row(Text("News disabled in replay", style="magenta"))
        for item in self.news[:8]:
            sent = float(item.get("sentiment") or 0)
            col = "green" if sent > 0 else "red" if sent < 0 else "white"
            mark = " NEW" if item.get("is_new") else ""
            when = item.get("when")
            hh = when.strftime("%H:%M") if when else ""
            news_tbl.add_row(Text(f"{hh}{mark} {item['title'][:70]}", style=col))
        return Panel(Group(t, news_tbl), title="Bias / Go-No-Go")

    def _setups_plan(self) -> Panel:
        tbl = Table(expand=True, box=None)
        tbl.add_column("setup")
        tbl.add_column("state")
        tbl.add_column("detail")
        for s in self.setups:
            st = s.get("state", "")
            col = "green" if st == "TRIGGERED" else "red" if st == "BLOCKED" else "dim"
            d = s.get("direction") or ""
            extra = ""
            if st == "TRIGGERED":
                extra = f"{d} {s.get('entry'):.0f} stop {s.get('stop'):.0f} tgt {s.get('target'):.0f}"
            tbl.add_row(s.get("name", ""), Text(st, style=col), (s.get("reason") or "") + " " + extra)
        lines = []
        if self.plan:
            p = self.plan
            tm_lab = (self.tm or {}).get("label", "")
            lines.append(f"{p['contract']}  {tm_lab}")
            lines.append(
                f"prem {p['entry_premium']:.2f}  stop {p['stop_premium']:.2f}  tgt {p['target_premium']:.2f}  "
                f"Δ {p['delta']:.2f} θ/d {p['theta_day']:.2f}"
            )
            if p.get("skip"):
                lines.append(p["skip_reason"])
            else:
                lines.append(
                    f"lots {p['lots']}  capital ₹{p['capital_needed']:.0f}  max loss ₹{p['max_loss']:.0f}  "
                    f"max profit ₹{p['max_profit']:.0f}"
                )
                lines.append(f"gross R:R {p['gross_rr']:.2f}  NET R:R {p['net_rr']:.2f} (after costs)")
                if p.get("breakeven_index"):
                    lines.append(f"index move to BE after costs ~ {p['breakeven_index']:.0f}")
            if p.get("expiry_afternoon_warn"):
                lines.append("WARN: expiry-day afternoon — gamma/theta risk is high.")
            if p.get("stt_flag"):
                lines.append("STT rates changed recently — verify costs.")
            lines.append("Partial at 1R then stop to cost.")
        else:
            lines.append("No triggered setup / no plan.")
        return Panel(Group(tbl, Text("\n".join(lines))), title="Setups / Trade plan")

    def _globals_events(self) -> Panel:
        g = Table(expand=True, box=None)
        g.add_column("ticker")
        g.add_column("%")
        for r in self.globals_rows:
            chg = r.get("chg_pct")
            col = "green" if (chg or 0) > 0 else "red" if (chg or 0) < 0 else "white"
            g.add_row(r["ticker"], Text("n/a" if chg is None else f"{chg:+.2f}%", style=col))
        ev = Table(title="Event log", expand=True, box=None)
        ev.add_column("e")
        for e in self.events[-10:]:
            ev.add_row(e)
        nxt = next_open(self.now(), self.holidays)
        cd = nxt - self.now()
        extra = Text(f"next open {nxt.strftime('%a %Y-%m-%d %H:%M')} ({cd})", style="dim")
        return Panel(Group(g, extra, ev), title="Global cues")

    def _tm_panel(self) -> Panel:
        tm = self.tm
        if not tm:
            body = "Time Machine idle (needs a triggered plan).\nHistory replay uses modeled premiums on the 59d cache."
            if self.hist_stats:
                bits = []
                for name, st in (self.hist_stats.get("setups") or {}).items():
                    warn = " n<30" if st.get("small_sample") else ""
                    bits.append(
                        f"{name}: fires {st['fires']} WR {st['win_rate']:.0%} avgR {st['avg_net_R']:+.2f}{warn}"
                    )
                body += "\n" + "\n".join(bits)
            return Panel(body, title="Time Machine")
        txt = (
            f"{tm.get('label')}  target-first {tm['target_first']:.0%}  stop-first {tm['stop_first']:.0%}  "
            f"time/neither {tm['neither']:.0%}\n"
            f"net P&L median ₹{tm['median_pnl']:.0f}  worst5% ₹{tm['worst_5']:.0f}  best5% ₹{tm['best_5']:.0f}  "
            f"EV ₹{tm['ev']:.0f}\n{tm.get('verdict')}"
        )
        return Panel(txt, title="Time Machine")

    def _paper_panel(self, now: datetime, phase: str, spot: float | None) -> Panel:
        if phase == "SQUARE_OFF":
            title = "SQUARE OFF — EXIT EVERYTHING"
            style = "bold white on red"
        else:
            title, style = "Paper position", "none"
        if not self.paper:
            return Panel("No open paper trade.", title=title, style=style)
        p = self.paper
        left = p.time_stop_minutes - (now - p.entry_time).total_seconds() / 60.0
        mark = None
        dist_s = dist_t = None
        net = None
        if spot is not None:
            try:
                mark = p.mark_premium(spot, now)
                _, _, net = p.net_if_exit(spot, now)
            except Exception:
                pass
            if p.direction == "long":
                dist_s = spot - p.stop_index
                dist_t = p.target_index - spot
            else:
                dist_s = p.stop_index - spot
                dist_t = spot - p.target_index
        txt = (
            f"{p.contract}  {p.setup} {p.direction}\n"
            f"entry {p.entry_premium:.2f}  mark {mark if mark is None else f'{mark:.2f}'}  "
            f"stop {p.stop_premium:.2f} tgt {p.target_premium:.2f}\n"
            f"qty {p.remaining_qty}/{p.qty}  modeled P&L ₹{0 if net is None else f'{net:.0f}'}\n"
            f"index dist stop {dist_s:.0f}  target {dist_t:.0f}   time-stop {left:.0f}m"
        )
        return Panel(txt, title=title, style=style)

    def _stats_panel(self, now: datetime, phase: str) -> Panel:
        st = day_stats(now.date())
        halt = self._halt(st)
        goal = float(self.cfg["daily_profit_goal"])
        lim = float(self.cfg["daily_loss_limit"])
        txt = (
            f"trades {st['trades']}/{self.cfg['max_trades_per_day']}  P&L ₹{st['pnl']:.0f}\n"
            f"goal ₹{goal:.0f} ({(st['pnl']/goal*100 if goal else 0):.0f}%)  "
            f"loss-limit ₹{lim:.0f}"
        )
        if halt:
            txt = "STOP TRADING\n" + txt + "\nNew TAKE labels blocked."
            return Panel(txt, title="Day stats", style="bold white on red")
        if phase == "POST_MARKET":
            txt += "\nSession review — journal stats above. Check tomorrow's events in config."
            for e in self.cfg.get("events", []):
                try:
                    d = date.fromisoformat(str(e.get("date")))
                except ValueError:
                    continue
                if d >= now.date():
                    txt += f"\n{e.get('date')} {e.get('name')}"
                    if d != now.date() and len(txt) > 400:
                        break
        if phase == "CLOSED":
            nxt = next_open(now, self.holidays)
            txt += f"\nMarket closed. Last session P&L ₹{st['pnl']:.0f}. Next open {nxt}."
        return Panel(txt, title="Day stats")


def _help_text() -> Text:
    return Text(
        "n/b/s select index  t run Time Machine  c live chart\n"
        "p enter live option premium (IV solve)  e log I entered  x log exit\n"
        "r replay YYYY-MM-DD  q quit\n"
        "Never places orders. Probabilities are resampled history, not predictions.\n"
        + SEBI_NOTE
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indian index options decision support (no broker).")
    p.add_argument("--replay", help="YYYY-MM-DD historical session")
    p.add_argument("--speed", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ensure_config()
    CONSOLE.print(
        "[bold yellow]WARNING:[/] NSE lot sizes may have been revised (circular dated 3 Oct 2026). "
        "Verify lot/step/expiry on NSE/BSE. STT and charges also change — edit config.json."
    )
    replay_day = date.fromisoformat(args.replay) if args.replay else None
    app = App(cfg, replay_day=replay_day, speed=args.speed)
    app.refresh_history()
    app.refresh_index()
    app.refresh_globals()
    app.refresh_news()
    app.recompute()
    try:
        with Live(app.render(), console=CONSOLE, refresh_per_second=8, screen=True) as live:
            while True:
                try:
                    ch = poll_key()
                    if ch:
                        if not app.handle_key(ch):
                            break
                    app.tick_tasks()
                    live.update(app.render())
                except KeyboardInterrupt:
                    break
                except Exception:
                    LOG.exception("loop")
                    time.sleep(1)
                time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    CONSOLE.print("Stopped. No orders were sent.")


if __name__ == "__main__":
    main()
