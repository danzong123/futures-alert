"""
Global news module — collect market context from multiple AKShare sources.
7-day rolling cache: headlines accumulate across scan cycles for medium/long-term
industry and macro news coverage, not just same-day flash headlines.
"""
import akshare as ak
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Rolling 7-day news cache (persists across scan cycles)
# Keyed by headline hash for dedup; stores headline + timestamp + source
# -----------------------------------------------------------
_NEWS_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_HOURS = 168  # 7 days * 24 hours

# Maximum cache size to prevent unbounded growth
_MAX_CACHE_SIZE = 500


def _prune_cache():
    """Remove entries older than TTL."""
    now = datetime.now()
    cutoff = now - timedelta(hours=_CACHE_TTL_HOURS)
    stale = [k for k, v in _NEWS_CACHE.items()
             if v.get("timestamp") and v["timestamp"] < cutoff]
    for k in stale:
        del _NEWS_CACHE[k]
    if stale:
        logger.debug("Pruned %d stale news entries from cache", len(stale))

    # Hard cap: if still too many, keep newest
    if len(_NEWS_CACHE) > _MAX_CACHE_SIZE:
        sorted_items = sorted(
            _NEWS_CACHE.items(),
            key=lambda x: x[1].get("timestamp", datetime.min),
            reverse=True,
        )
        for k, _ in sorted_items[_MAX_CACHE_SIZE:]:
            del _NEWS_CACHE[k]


def _add_headlines(headlines: List[str], source: str = ""):
    """Add new headlines to cache. Deduplicates by headline text."""
    now = datetime.now()
    added = 0
    for h in headlines:
        if not h or not isinstance(h, str):
            continue
        h = h.strip()[:300]  # truncate long headlines
        key = h  # use headline text as key for natural dedup
        if key not in _NEWS_CACHE:
            _NEWS_CACHE[key] = {
                "headline": h,
                "timestamp": now,
                "source": source,
            }
            added += 1
    if added:
        logger.debug("Added %d new headlines from %s", added, source)


def collect_global_context() -> Dict:
    """
    Collect global market context from multiple sources.
    Returns all cached headlines from the last 7 days, plus fresh headlines
    from current cycle's API calls.
    """
    with _CACHE_LOCK:
        _prune_cache()

    fresh_headlines: List[str] = []

    # Source 1: Tonghuashun global flash news (primary, real-time)
    try:
        news_df = ak.stock_info_global_ths()
        if not news_df.empty and "title" in news_df.columns:
            titles = news_df["title"].dropna().head(15).tolist()
            with _CACHE_LOCK:
                _add_headlines(titles, source="ths_flash")
            fresh_headlines.extend(titles)
            logger.debug("THS flash news: %d headlines", len(titles))
    except Exception as e:
        logger.debug("THS flash news failed: %s", str(e)[:80])

    # Source 2: East Money global news (supplementary, broader coverage)
    try:
        em_news = ak.stock_info_global_em()
        if not em_news.empty:
            title_col = None
            for col in ["title", "content", "art_title"]:
                if col in em_news.columns:
                    title_col = col
                    break
            if title_col:
                titles = em_news[title_col].dropna().head(10).tolist()
                with _CACHE_LOCK:
                    _add_headlines(titles, source="em_global")
                fresh_headlines.extend(titles)
                logger.debug("EM global news: %d headlines", len(titles))
    except Exception as e:
        logger.debug("EM global news failed: %s", str(e)[:80])

    # Source 3: Tonghuashun futures-specific news (if available)
    try:
        futures_news = ak.futures_news_ths()
        if not futures_news.empty:
            title_col = None
            for col in ["title", "content"]:
                if col in futures_news.columns:
                    title_col = col
                    break
            if title_col:
                titles = futures_news[title_col].dropna().head(10).tolist()
                with _CACHE_LOCK:
                    _add_headlines(titles, source="ths_futures")
                fresh_headlines.extend(titles)
                logger.debug("THS futures news: %d headlines", len(titles))
    except Exception as e:
        logger.debug("THS futures news failed: %s", str(e)[:80])

    # Collect all cached headlines within the 7-day window
    now = datetime.now()
    cutoff = now - timedelta(hours=_CACHE_TTL_HOURS)
    with _CACHE_LOCK:
        window_headlines: List[dict] = []
        for entry in _NEWS_CACHE.values():
            ts = entry.get("timestamp")
            if ts and ts >= cutoff:
                window_headlines.append(entry)
        # Sort newest first
        window_headlines.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        cache_size = len(_NEWS_CACHE)

    headlines_text = [e["headline"] for e in window_headlines]
    # Deduplicate final list while preserving order
    seen = set()
    unique_headlines = []
    for h in headlines_text:
        if h not in seen:
            seen.add(h)
            unique_headlines.append(h)

    fresh_count = len(set(fresh_headlines))
    logger.info(
        "News cache: %d total, %d in 7d window (fresh this cycle: %d)",
        cache_size, len(unique_headlines), fresh_count,
    )

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M"),
        "news_headlines": unique_headlines,
        "news_entries": window_headlines,  # full entries with timestamps
        "summary": f"7-day window: {len(unique_headlines)} headlines (cache: {cache_size})",
    }


def get_news_stats() -> Dict:
    """Return cache statistics for monitoring."""
    with _CACHE_LOCK:
        return {
            "cache_size": len(_NEWS_CACHE),
            "cache_ttl_hours": _CACHE_TTL_HOURS,
            "max_size": _MAX_CACHE_SIZE,
        }
