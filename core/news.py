from __future__ import annotations

import hashlib
import logging
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from zoneinfo import ZoneInfo

import feedparser

from core.config import ensure_lexicon
from core.data import IST

LOG = logging.getLogger("trade")

GOOGLE = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
EXTRA_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]


def _to_ist(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def _parse_time(entry: dict[str, Any]) -> datetime | None:
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return _to_ist(parsedate_to_datetime(str(raw)))
        except (TypeError, ValueError, OverflowError):
            pass
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
        except (TypeError, ValueError):
            return None
    return None


def _hid(title: str, link: str) -> str:
    return hashlib.sha1(f"{title}|{link}".encode("utf-8", errors="ignore")).hexdigest()[:16]


def _score_text(text: str, lexicon: dict[str, Any]) -> tuple[float, list[str], bool]:
    low = text.lower()
    score = 0.0
    tags: list[str] = []
    for word, w in lexicon.get("positive", {}).items():
        if word.lower() in low:
            score += float(w)
    for word, w in lexicon.get("negative", {}).items():
        if word.lower() in low:
            score -= float(w)
    for tag, keys in lexicon.get("tags", {}).items():
        if any(k.lower() in low for k in keys):
            tags.append(tag)
    moving = False
    for kw in lexicon.get("market_moving", []):
        if str(kw).lower() in low:
            moving = True
            tags.append(f"ALERT:{kw}")
    return score, tags, moving


def _recency_weight(when: datetime | None, now: datetime) -> float:
    if when is None:
        return 0.35
    hours = max(0.0, (now - when).total_seconds() / 3600.0)
    return pow(2.0, -hours / 6.0)


def fetch_feed(url: str) -> list[dict[str, Any]]:
    parsed = feedparser.parse(url)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        raise RuntimeError(f"feed failed {url}: {getattr(parsed, 'bozo_exception', '')}")
    out = []
    for e in parsed.entries[:25]:
        title = str(e.get("title") or "").strip()
        link = str(e.get("link") or "").strip()
        if not title:
            continue
        out.append({"title": title, "link": link, "when": _parse_time(e), "source": url})
    return out


def collect_news(cfg: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    lexicon = ensure_lexicon()
    items: list[dict[str, Any]] = []
    for q in cfg.get("news_queries", []):
        url = GOOGLE.format(q=urllib.parse.quote(str(q)))
        try:
            items.extend(fetch_feed(url))
        except Exception:
            LOG.exception("news query failed %s", q)
    for url in EXTRA_FEEDS:
        try:
            items.extend(fetch_feed(url))
        except Exception:
            LOG.debug("optional feed skipped %s", url, exc_info=True)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in items:
        hid = _hid(it["title"], it.get("link") or "")
        if hid in seen:
            continue
        seen.add(hid)
        sent, tags, moving = _score_text(it["title"], lexicon)
        w = _recency_weight(it.get("when"), now)
        it["id"] = hid
        it["sentiment"] = sent
        it["tags"] = tags
        it["moving"] = moving
        it["weight"] = w
        it["weighted"] = sent * w
        it["when_ist"] = it.get("when")
        deduped.append(it)
    deduped.sort(key=lambda x: x.get("when") or datetime(1970, 1, 1, tzinfo=IST), reverse=True)
    return deduped


def news_score(items: list[dict[str, Any]]) -> float:
    """Aggregate headline score in [-100, 100]. Input to bias, never a forecast."""
    if not items:
        return 0.0
    top = items[:30]
    num = sum(float(i.get("weighted") or 0.0) for i in top)
    den = sum(float(i.get("weight") or 0.0) for i in top) or 1.0
    avg = num / den
    return max(-100.0, min(100.0, avg * 12.0))


def mark_new(items: list[dict[str, Any]], seen_ids: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    new_ids = {i["id"] for i in items}
    for i in items:
        i["is_new"] = i["id"] not in seen_ids
    return items, new_ids


def market_moving(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if i.get("moving") and i.get("is_new")]
