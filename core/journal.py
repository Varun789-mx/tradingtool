from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.data import IST
from core.options import bs_price, round_trip_costs

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "journal.csv"
COLUMNS = [
    "date",
    "index",
    "setup",
    "direction",
    "contract",
    "entry_time",
    "entry_premium",
    "exit_time",
    "exit_premium",
    "qty",
    "gross",
    "costs",
    "net",
    "reason",
]


def _ensure() -> None:
    if not JOURNAL.exists():
        with JOURNAL.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def append_row(row: dict[str, Any]) -> None:
    _ensure()
    with JOURNAL.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def load_rows() -> list[dict[str, str]]:
    _ensure()
    with JOURNAL.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def day_stats(day: date) -> dict[str, Any]:
    rows = [r for r in load_rows() if str(r.get("date")) == day.isoformat()]
    closed = [r for r in rows if r.get("exit_time")]
    nets = []
    for r in closed:
        try:
            nets.append(float(r.get("net") or 0))
        except ValueError:
            nets.append(0.0)
    pnl = sum(nets)
    return {
        "trades": len(closed),
        "open_rows": len(rows) - len(closed),
        "pnl": pnl,
        "rows": rows,
    }


@dataclass
class PaperPosition:
    index_name: str
    setup: str
    direction: str
    contract: str
    option_type: str
    strike: float
    expiry: date
    iv: float
    rate: float
    lot: int
    lots: int
    qty: int
    remaining_qty: int
    entry_time: datetime
    entry_index: float
    entry_premium: float
    stop_index: float
    target_index: float
    partial_index: float
    stop_premium: float
    target_premium: float
    exchange: str
    costs_cfg: dict[str, Any]
    time_stop_minutes: int
    partial_done: bool = False
    realized_gross: float = 0.0
    realized_costs: float = 0.0
    events: list[str] = field(default_factory=list)

    def mark_premium(self, spot: float, now: datetime) -> float:
        from core.options import years_to_expiry

        t = years_to_expiry(now, self.expiry)
        return bs_price(spot, float(self.strike), max(t, 1e-8), self.rate, self.iv, self.option_type)

    def unrealized(self, spot: float, now: datetime) -> float:
        mark = self.mark_premium(spot, now)
        return (mark - self.entry_premium) * self.remaining_qty

    def net_if_exit(self, spot: float, now: datetime) -> tuple[float, float, float]:
        mark = self.mark_premium(spot, now)
        qty = self.remaining_qty
        gross = (mark - self.entry_premium) * qty + self.realized_gross
        c = round_trip_costs(
            self.entry_premium,
            mark,
            qty,
            self.costs_cfg,
            self.exchange,
            n_entry_orders=1,
            n_exit_orders=1 + (1 if self.partial_done else 0),
        )
        costs = float(c["total"]) + self.realized_costs
        return gross, costs, gross - costs

    def manage(self, bar: Any, now: datetime, hard_exit: bool) -> dict[str, Any] | None:
        """Check stop/target/partial/time/hard using HIGH/LOW of the monitoring bar. Returns exit dict or None."""
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])
        elapsed = (now - self.entry_time).total_seconds() / 60.0
        reason = None
        exit_index = close
        if hard_exit:
            reason, exit_index = "hard_exit", close
        elif elapsed >= self.time_stop_minutes:
            reason, exit_index = "time_stop", close
        elif self.direction == "long":
            if low <= self.stop_index:
                reason, exit_index = "stop", self.stop_index
            elif high >= self.target_index:
                reason, exit_index = "target", self.target_index
            elif not self.partial_done and high >= self.partial_index:
                self._partial(self.partial_index, now, "partial_1R")
                return None
        else:
            if high >= self.stop_index:
                reason, exit_index = "stop", self.stop_index
            elif low <= self.target_index:
                reason, exit_index = "target", self.target_index
            elif not self.partial_done and low <= self.partial_index:
                self._partial(self.partial_index, now, "partial_1R")
                return None
        if reason is None:
            return None
        return self.close(exit_index, now, reason)

    def _partial(self, index_level: float, now: datetime, why: str) -> None:
        if self.remaining_qty <= 1:
            return
        half = (self.remaining_qty // self.lot // 2) * self.lot
        if half <= 0:
            half = self.lot if self.remaining_qty > self.lot else 0
        if half <= 0:
            return
        from core.options import years_to_expiry

        t = years_to_expiry(now, self.expiry)
        px = bs_price(index_level, float(self.strike), max(t, 1e-8), self.rate, self.iv, self.option_type)
        gross = (px - self.entry_premium) * half
        c = round_trip_costs(self.entry_premium, px, half, self.costs_cfg, self.exchange, 1, 1)
        # Stop to cost: remaining stop = entry index
        self.stop_index = self.entry_index
        self.realized_gross += gross
        self.realized_costs += float(c["total"]) * 0.5
        self.remaining_qty -= half
        self.lots = max(1, self.remaining_qty // self.lot)
        self.partial_done = True
        self.events.append(why)
        self.stop_premium = self.entry_premium

    def close(self, index_level: float, now: datetime, reason: str) -> dict[str, Any]:
        from core.options import years_to_expiry

        t = years_to_expiry(now, self.expiry)
        px = bs_price(index_level, float(self.strike), max(t, 1e-8), self.rate, self.iv, self.option_type)
        qty = self.remaining_qty
        gross = (px - self.entry_premium) * qty + self.realized_gross
        c = round_trip_costs(
            self.entry_premium,
            px,
            max(qty, 1),
            self.costs_cfg,
            self.exchange,
            1,
            1 + (1 if self.partial_done else 0),
        )
        costs = float(c["total"]) + self.realized_costs
        net = gross - costs
        row = {
            "date": self.entry_time.astimezone(IST).date().isoformat(),
            "index": self.index_name,
            "setup": self.setup,
            "direction": self.direction,
            "contract": self.contract,
            "entry_time": self.entry_time.astimezone(IST).strftime("%H:%M:%S"),
            "entry_premium": f"{self.entry_premium:.2f}",
            "exit_time": now.astimezone(IST).strftime("%H:%M:%S"),
            "exit_premium": f"{px:.2f}",
            "qty": str(self.qty),
            "gross": f"{gross:.2f}",
            "costs": f"{costs:.2f}",
            "net": f"{net:.2f}",
            "reason": reason,
        }
        append_row(row)
        self.remaining_qty = 0
        return {"row": row, "reason": reason, "net": net, "exit_premium": px, "exit_index": index_level}


def position_from_plan(plan: dict[str, Any], setup: dict[str, Any], now: datetime, cfg: dict[str, Any]) -> PaperPosition:
    return PaperPosition(
        index_name=plan["index"],
        setup=setup.get("name", ""),
        direction=plan["direction"],
        contract=plan["contract"],
        option_type=plan["option_type"],
        strike=float(plan["strike"]),
        expiry=plan["expiry"],
        iv=float(plan["iv"]),
        rate=float(cfg["risk_free"]),
        lot=int(plan["lot_size"]),
        lots=int(plan["lots"]),
        qty=int(plan["qty"]),
        remaining_qty=int(plan["qty"]),
        entry_time=now,
        entry_index=float(plan["entry_index"]),
        entry_premium=float(plan["entry_premium"]),
        stop_index=float(plan["stop_index"]),
        target_index=float(plan["target_index"]),
        partial_index=float(setup.get("partial") or plan["entry_index"]),
        stop_premium=float(plan["stop_premium"]),
        target_premium=float(plan["target_premium"]),
        exchange=str(plan["exchange"]),
        costs_cfg=cfg["costs"],
        time_stop_minutes=int(cfg["time_stop_minutes"]),
    )
