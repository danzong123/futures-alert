"""
Simplified futures alert scheduler.

Core flow: fetch quotes -> load K-lines -> build snapshots -> detect signals -> push.
  - Per-contract error isolation: one failure never halts the cycle.
  - Anti-spam: one alert per symbol per cooldown window.
  - All parameters locked in config.yaml; no auto-modification at runtime.
"""
import os, sys, time, yaml, logging, argparse
import pandas as pd
from datetime import datetime, time as dt_time

# Strip broken proxy settings that would block all outbound HTTP requests
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_key, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import get_main_contract_quotes, get_minute_candles, get_daily_candles
from src.indicators import calc_all_indicators
from src.strategy import analyze_contracts_simple
from src.notifier import WeChatNotifier, PushDispatcher
from src.database import (init_db, save_signals, save_alert, save_contract_snapshots_batch,
                         save_verification_result, get_verification_summary)
from src.global_news import collect_global_context
from src.news_analyzer import analyze_news_impact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "app.log"),
            encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
CACHE = {}          # symbol -> DataFrame (15-min K-line + indicators)
SWING_CACHE = {}    # symbol -> {high, low, last_date}  daily swing range, fixed intraday
ANTI_SPAM = {}      # symbol:signal_type -> last_alert_timestamp (epoch seconds, band reversal per-direction cooldown)
DAILY_STATS = {"date": "", "pushed": 0}  # daily trade counter for swing trading limit
_LAST_SWING_UPDATE_DAY = ""  # Track which date we last fully updated swing ranges (once per day, post-close only)


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def is_trading_time():
    now = datetime.now()
    t = now.time()
    if now.weekday() >= 5:
        return False
    day_s, day_e = dt_time(8, 55), dt_time(15, 0)
    night_s = dt_time(21, 0)
    night_end = dt_time(2, 30)
    if day_s <= t <= day_e:
        return True
    if t >= night_s or t <= night_end:
        return True
    return False


# ================================================================
# K-line loading: one contract at a time, full error isolation
# ================================================================

def load_all_klines(symbols: list, period: str, tail: int):
    """Load K-line data for every symbol. Already-cached symbols are refreshed.
    Returns (success_count, skip_count)."""
    ok_count = 0
    fail_count = 0
    for i, sym in enumerate(symbols):
        try:
            df = get_minute_candles(sym, period)
            if df is None or df.empty:
                fail_count += 1
                continue
            df = df.tail(tail).copy()
            df = calc_all_indicators(df)
            CACHE[sym] = df
            ok_count += 1
        except Exception as e:
            logger.warning("K-line skip %s: %s", sym, str(e)[:100])
            fail_count += 1
        # Progress log every 20 contracts
        if (i + 1) % 20 == 0:
            logger.info("  K-line progress: %d/%d (ok=%d fail=%d)",
                        i + 1, len(symbols), ok_count, fail_count)
    return ok_count, fail_count


# ================================================================
# Contract snapshot builder
# ================================================================

# ================================================================
# Swing range: daily K-line based, fixed intraday
# ================================================================

def update_swing_ranges(all_symbols: list, config: dict):
    """Load daily K-lines and update SWING_CACHE with swing high/low,
    daily-based indicators (Bollinger, ATR, RSI, MACD, MA), volume ratio,
    and OI change. Only refreshes when a new daily bar closes.
    Returns (updated_count, total_count)."""
    global _LAST_SWING_UPDATE_DAY
    today_str = datetime.now().strftime("%Y%m%d")

    # --- Time gate: only recalculate daily bands once per day, post-close ---
    # During trading hours (9:00-15:00), SWING_CACHE is read-only.
    # Full recalculation happens once after 15:00 close to reduce backend load.
    now = datetime.now()
    post_close = now.hour >= 15
    already_updated_today = (_LAST_SWING_UPDATE_DAY == today_str)

    # Cold start: populate cache on first run regardless of time
    if not SWING_CACHE:
        logger.info("Swing cache cold start - initializing daily band data")
    elif already_updated_today:
        logger.debug("Swing ranges: already updated for %s (read-only intraday)", today_str)
        return 0, len(all_symbols)
    elif not post_close:
        if SWING_CACHE:
            logger.debug("Swing ranges: trading hours read-only, %d cached", len(SWING_CACHE))
        return 0, len(all_symbols)

    # --- Post-close first update today: recalculate all daily bands ---
    logger.info("Swing ranges: post-close recalculation for %s (%d symbols)", today_str, len(all_symbols))

    data_cfg = config.get("data_table", {})
    daily_bars = int(data_cfg.get("swing_daily_bars", 20))
    updated = 0

    for sym in all_symbols:
        try:
            df = get_daily_candles(sym)
            if df is None or df.empty or "high" not in df.columns or "low" not in df.columns:
                continue

            # Use enough bars for indicators (need 60+ for reliable daily MA/ATR/BOLL)
            work_df = df.tail(max(daily_bars, 80))
            if len(work_df) < 10:
                continue

            # Only refresh when a new daily bar closes (date changes)
            if "date" in work_df.columns:
                latest_bar_date = str(work_df["date"].iloc[-1].date()).replace("-", "")
            else:
                latest_bar_date = today_str
            cached = SWING_CACHE.get(sym)
            if cached and cached.get("last_date") == latest_bar_date:
                continue

            # Compute all daily indicators
            ind_df = calc_all_indicators(work_df)
            latest = ind_df.iloc[-1]
            prev = ind_df.iloc[-2] if len(ind_df) >= 2 else None

            # Swing high/low from last N daily bars
            recent_n = ind_df.tail(daily_bars)
            swing_high = float(recent_n["high"].max())
            swing_low = float(recent_n["low"].min())

            # Helper: safe float extraction from indicator row
            def _d(col, decimals=4):
                v = latest.get(col)
                if v is None or (isinstance(v, float) and v != v):
                    return None
                return round(float(v), decimals)

            # Daily volume ratio: latest / 20-day avg
            vol_ratio = None
            if "volume" in recent_n.columns:
                avg_vol = float(recent_n["volume"].mean())
                latest_vol = float(recent_n["volume"].iloc[-1])
                if avg_vol > 0:
                    vol_ratio = round(latest_vol / avg_vol, 1)

            # Daily OI change
            oi_change_pct = None
            if "open_interest" in work_df.columns and len(work_df) >= 2:
                prev_oi = float(work_df["open_interest"].iloc[-2])
                latest_oi = float(work_df["open_interest"].iloc[-1])
                if prev_oi > 0 and not pd.isna(prev_oi) and not pd.isna(latest_oi):
                    oi_change_pct = round((latest_oi - prev_oi) / prev_oi * 100, 1)

            # MACD bar comparison (current vs previous daily bar)
            macd_bar = _d("macd_bar", 4)
            macd_bar_prev = round(float(prev["macd_bar"]), 4) if prev is not None and "macd_bar" in prev and not pd.isna(prev["macd_bar"]) else None

            SWING_CACHE[sym] = {
                "high": swing_high,
                "low": swing_low,
                "last_date": latest_bar_date,
                "vol_ratio": vol_ratio,
                "oi_change_pct": oi_change_pct,
                # Daily Bollinger (20-day, 2x std)
                "boll_upper": _d("boll_upper", 2),
                "boll_mid": _d("boll_mid", 2),
                "boll_lower": _d("boll_lower", 2),
                # Daily ATR (14-day)
                "atr": _d("atr", 2),
                # Daily RSI (14-day)
                "rsi": _d("rsi", 1),
                # Daily MACD
                "macd_bar": macd_bar,
                "macd_bar_prev": macd_bar_prev,
                "macd_dif": _d("macd_dif", 4),
                "macd_dea": _d("macd_dea", 4),
                # Daily MA
                "ma5": _d("ma5", 2),
                "ma10": _d("ma10", 2),
                "ma20": _d("ma20", 2),
            }
            updated += 1
        except Exception as e:
            logger.debug("Swing range skip %s: %s", sym, str(e)[:80])

    # Mark today's update as complete (prevents re-update until next day post-close)
    _LAST_SWING_UPDATE_DAY = today_str

    return updated, len(all_symbols)

def build_contract_snapshots(quotes_df, news_impacts, config):
    """Build core data records for each contract from cached K-lines."""
    data_cfg = config.get("data_table", {})
    range_period = data_cfg.get("range_period", 20)
    core_indicators = data_cfg.get("core_indicators", ["rsi", "macd_bar"])

    # Build per-symbol news map
    news_map = {}
    for ni in news_impacts:
        for sym in ni.get("symbols", []):
            if sym not in news_map:
                news_map[sym] = []
            headline = ni.get("headline", "")
            hl_lower = headline.lower()
            if any(kw in hl_lower for kw in ["\u6da8", "rise", "bull", "surge", "\u5229\u591a", "\u5927\u6da8"]):
                direction = "bullish"
            elif any(kw in hl_lower for kw in ["\u8dcc", "fall", "bear", "crash", "\u5229\u7a7a", "\u66b4\u8dcc"]):
                direction = "bearish"
            else:
                direction = "neutral"
            news_map[sym].append({"headline": headline, "direction": direction})

    records = []
    for _, row in quotes_df.iterrows():
        symbol = row.get("symbol", "")
        if not symbol or symbol not in CACHE:
            continue

        candle_df = CACHE[symbol]
        if candle_df.empty or len(candle_df) < range_period:
            continue

        name = str(row.get("name", symbol))

        # Swing range: use daily K-line based SWING_CACHE (fixed intraday)
        swing = SWING_CACHE.get(symbol)
        if swing and swing.get("high") and swing.get("low"):
            range_high = swing["high"]
            range_low = swing["low"]
        else:
            # Fallback: compute from 15-min K-line (should rarely happen)
            recent = candle_df.tail(range_period)
            range_high = float(recent["high"].max()) if "high" in recent.columns else None
            range_low = float(recent["low"].min()) if "low" in recent.columns else None

        # Real-time price (may be NaN outside trading hours; fall back to latest K-line close)
        rt = row.get("last_price")
        if rt is not None and not (isinstance(rt, float) and rt != rt):
            latest_price = float(rt)
        else:
            latest_price = float(candle_df.iloc[-1]["close"]) if "close" in candle_df.columns else None

        # Core indicators: use daily-based values from SWING_CACHE
        ind1_name, ind1_value = None, None
        ind2_name, ind2_value = None, None
        if len(core_indicators) >= 1 and swing:
            i1 = core_indicators[0]
            v = swing.get(i1)
            if v is not None:
                ind1_name, ind1_value = i1, round(float(v), 2)
        if len(core_indicators) >= 2 and swing:
            i2 = core_indicators[1]
            v = swing.get(i2)
            if v is not None:
                ind2_name, ind2_value = i2, round(float(v), 2)

        # Extended indicators (daily-based from SWING_CACHE; fallback to 15-min)
        def _safe_float(col):
            row = candle_df.iloc[-1]
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and v != v):
                return round(float(v), 4)
            return None

        def _swing(key, decimals=4):
            """Get a value from SWING_CACHE, falling back to 15-min indicator."""
            if swing and swing.get(key) is not None:
                v = swing[key]
                return round(float(v), decimals) if v is not None else None
            return _safe_float(key)

        atr = _swing("atr", 4)
        boll_upper = _swing("boll_upper", 4)
        boll_mid = _swing("boll_mid", 4)
        boll_lower = _swing("boll_lower", 4)
        volume_ratio = _swing("vol_ratio", 4) if swing and swing.get("vol_ratio") is not None else _safe_float("volume_ratio")
        # Compute position_pct from swing range: (price - low) / (high - low) * 100
        if range_high and range_low and range_high > range_low and latest_price:
            position_pct = round((latest_price - range_low) / (range_high - range_low) * 100, 0)
        else:
            position_pct = _safe_float("position_pct")
        ma5 = _swing("ma5", 4)
        ma10 = _swing("ma10", 4)
        ma20 = _swing("ma20", 4)
        oi_change_pct = _swing("oi_change_pct", 4) if swing and swing.get("oi_change_pct") is not None else _safe_float("oi_change_pct")

        # News attachment (all relevant headlines with impact duration)
        sym_news = news_map.get(symbol, [])
        # Sort: medium-term (中期) first, then long (长期), then short (短期)
        sym_news.sort(key=lambda n: {"medium": 0, "long": 1, "short": 2}.get(n.get("impact", "short"), 2))
        news_summary = sym_news[0]["headline"][:200] if sym_news else None
        news_direction = sym_news[0]["direction"] if sym_news else None
        news_impact_label = sym_news[0].get("impact_label", "") if sym_news else ""

        records.append({
            "symbol": symbol,
            "name": name,
            "range_high": range_high,
            "range_low": range_low,
            "latest_price": latest_price,
            "indicator_1_name": ind1_name,
            "indicator_1_value": ind1_value,
            "indicator_2_name": ind2_name,
            "indicator_2_value": ind2_value,
            "atr": atr,
            "boll_upper": boll_upper,
            "boll_mid": boll_mid,
            "boll_lower": boll_lower,
            "volume_ratio": volume_ratio,
            "position_pct": position_pct,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "oi_change_pct": oi_change_pct,
            "macd_bar_prev": _swing("macd_bar_prev", 4) if swing and swing.get("macd_bar_prev") is not None else None,
            "news_summary": news_summary,
            "news_direction": news_direction,
            "news_impact_label": news_impact_label,
            "all_news": sym_news,  # all matched news for this symbol with duration
        })

    return records


# ================================================================
# Main check cycle
# ================================================================

def run_check(config, notifier):
    logger.info("=" * 50)
    logger.info("Starting signal check...")

    # --- Step 1: Fetch main contract quotes (single lightweight API call) ---
    quotes = get_main_contract_quotes()
    if quotes.empty:
        logger.warning("No quotes data - skipping cycle")
        return 0

    all_symbols = quotes["symbol"].tolist()
    total_count = len(all_symbols)
    logger.info("Main contracts: %d total", total_count)

    # --- Watchlist filter: only monitor configured symbols (swing trading mode) ---
    watchlist = config.get("watchlist", [])
    if watchlist:
        quotes = quotes[quotes["symbol"].isin(watchlist)]
        all_symbols = quotes["symbol"].tolist()
        logger.info("Watchlist filter: %d/%d contracts selected", len(all_symbols), total_count)
        total_count = len(all_symbols)
    # Load K-lines first, then filter by which ones actually have recent data

    # --- Step 2: Load K-lines for all contracts (per-contract error isolation) ---
    monitor_cfg = config.get("monitor", {})
    period = str(monitor_cfg.get("kline_period", "5"))
    tail = int(monitor_cfg.get("kline_tail", 60))

    ok, fail = load_all_klines(all_symbols, period, tail)
    logger.info("K-line load: %d ok, %d failed (cache: %d total)",
                ok, fail, len(CACHE))

    # Filter to active contracts: K-line data exists AND latest bar is within 4 hours
    active_symbols = []
    cutoff = datetime.now().timestamp() - 4 * 3600
    for sym in list(CACHE.keys()):
        df = CACHE[sym]
        if df is not None and not df.empty and "datetime" in df.columns:
            latest_dt = df["datetime"].iloc[-1]
            if hasattr(latest_dt, 'timestamp'):
                if latest_dt.timestamp() > cutoff:
                    active_symbols.append(sym)
            else:
                active_symbols.append(sym)  # can't check, include anyway
        else:
            active_symbols.append(sym)  # no datetime col, include
    # Remove inactive symbols from cache
    for sym in list(CACHE.keys()):
        if sym not in active_symbols:
            del CACHE[sym]
    # Also filter quotes to match
    quotes = quotes[quotes["symbol"].isin(active_symbols)]
    all_symbols = active_symbols
    logger.info("Active (recent K-line): %d / %d total", len(all_symbols), total_count)

    if len(CACHE) == 0:
        logger.warning("No K-line data available - skipping analysis")
        return 0

    # --- Step 3: Build contract snapshots ---
    global_ctx = collect_global_context()
    headlines = global_ctx.get("news_headlines", [])
    news_impacts = analyze_news_impact(headlines)
    logger.info("News impact: %d matched items", len(news_impacts))

    # --- Step 3b: Update swing ranges from daily K-lines (fixed intraday) ---
    swing_updated, swing_total = update_swing_ranges(all_symbols, config)
    logger.info("Swing ranges: %d updated / %d total (cache: %d)",
                swing_updated, swing_total, len(SWING_CACHE))

    snapshots = build_contract_snapshots(quotes, news_impacts, config)
    if snapshots:
        save_contract_snapshots_batch(snapshots)
        logger.info("Core data snapshots saved: %d contracts", len(snapshots))

    # --- Step 4: Signal detection ---
    strategy_cfg = config.get("strategy", {})
    signals = analyze_contracts_simple(snapshots, ANTI_SPAM, strategy_cfg)
    logger.info("Signals detected: %d", len(signals))

    if signals:
        save_signals(signals)
        notify_cfg = config.get("notify", {})
        max_send = notify_cfg.get("max_alerts_per_batch", 5)
        # --- Daily trade limit enforcement (swing trading: max 3/day) ---
        trade_limits = config.get("trade_limits", {})
        daily_max = trade_limits.get("daily_max_alerts", 3)
        today_str = datetime.now().strftime("%Y%m%d")
        if DAILY_STATS.get("date") != today_str:
            DAILY_STATS["date"] = today_str
            DAILY_STATS["pushed"] = 0
            logger.info("Daily trade counter reset for %s", today_str)
        remaining = daily_max - DAILY_STATS["pushed"]
        if remaining <= 0:
            logger.info("Daily limit reached (%d/%d) - all alerts suppressed", DAILY_STATS["pushed"], daily_max)
            enqueued = 0
        else:
            batch_limit = min(max_send, remaining)
            pushed_count = 0
            for s in signals[:batch_limit]:
                save_alert(s["name"], s["message"])
                title = "Futures %s %s" % (s["name"], s["signal"])
                notifier.enqueue(title, s["message"])
                pushed_count += 1
            DAILY_STATS["pushed"] += pushed_count
            enqueued = pushed_count
            logger.info("Enqueued %d alert(s) | Daily: %d/%d", pushed_count, DAILY_STATS["pushed"], daily_max)
        longs = sum(1 for s in signals if s["signal"] == "long")
        shorts = sum(1 for s in signals if s["signal"] == "short")
        logger.info("Summary: LONG %d | SHORT %d", longs, shorts)
    else:
        logger.info("No signals - market neutral")

    return len(signals)


# ================================================================
# Main entry point
# ================================================================

# ================================================================
# Signal verification: check pushed signals against actual outcome
# ================================================================

def run_verify(config):
    """Verify signal accuracy by comparing predicted direction with subsequent price movement.
    For each signal: fetch current K-line, compare signal price to latest close.
    Bull correct if close > signal_price. Bear correct if close < signal_price."""
    from src.database import get_signal_history, get_connection
    import pandas as pd

    logger.info("=" * 50)
    logger.info("Signal Verification Mode")
    logger.info("=" * 50)

    # Get config
    monitor_cfg = config.get("monitor", {})
    period = str(monitor_cfg.get("kline_period", "5"))

    # Clear today's verification records before re-verifying
    try:
        conn = get_connection()
        today = datetime.now().strftime("%Y-%m-%d")
        conn.execute("DELETE FROM verification_log WHERE verify_date = ?", (today,))
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Load signals from today
    signals_df = get_signal_history(hours=48)
    if signals_df.empty:
        logger.info("No signals to verify")
        return

    # Group by symbol, take the latest signal per symbol
    signals_df["dt"] = pd.to_datetime(signals_df["timestamp"])
    signals_df = signals_df.sort_values("dt")
    # Keep all signals for verification (not just latest per symbol)

    total = len(signals_df)
    verified = 0
    correct = 0
    wrong = 0
    flat = 0
    skipped = 0

    # Report lines
    report = []
    report.append("")
    report.append("=" * 60)
    report.append("  SIGNAL VERIFICATION REPORT")
    report.append("=" * 60)
    report.append(f"  Period: last 48 hours | Total signals: {total}")
    report.append("-" * 60)

    for _, row in signals_df.iterrows():
        symbol = row["symbol"]
        name = row["name"]
        signal = row["signal"]
        signal_price = row.get("price", 0)
        signal_time = str(row["timestamp"])

        if signal_price is None or pd.isna(signal_price) or signal_price == 0:
            skipped += 1
            continue

        try:
            df = get_minute_candles(symbol, period)
            if df is None or df.empty:
                skipped += 1
                continue

            latest_close = float(df.iloc[-1]["close"])
            if pd.isna(latest_close):
                skipped += 1
                continue

            change_pct = round((latest_close - signal_price) / signal_price * 100, 2)
            verified += 1

            if signal == "long":
                is_correct = latest_close > signal_price
            else:
                is_correct = latest_close < signal_price

            # Classify: correct, wrong, or flat (no meaningful movement)
            if abs(change_pct) < 0.005:
                flat += 1
                status = "∼ FLAT"
            elif is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                wrong += 1
                status = "✗ WRONG"

            # Persist to database for daily review
            try:
                save_verification_result(
                    signal_id=int(row["id"]), symbol=symbol, name=name,
                    signal=signal, score=int(row.get("score", 0) or 0),
                    signal_price=signal_price, verify_price=latest_close,
                    change_pct=change_pct,
                    is_correct=is_correct, is_flat=(abs(change_pct) < 0.005),
                )
            except Exception:
                pass

            sig_display = signal.upper()
            if sig_display == "BULL":
                sig_display = "LONG"
            elif sig_display == "BEAR":
                sig_display = "SHORT"
            report.append(
                f"  {status:12s} | {name:10s} {symbol:6s} | "
                f"{sig_display:4s} | signal@{signal_price:.0f} -> "
                f"close@{latest_close:.0f} ({change_pct:+.2f}%)"
            )

        except Exception as e:
            logger.debug("Verify skip %s: %s", symbol, str(e)[:60])
            skipped += 1

    # Summary
    meaningful = correct + wrong
    accuracy = round(correct / meaningful * 100, 1) if meaningful > 0 else 0
    raw_accuracy = round(correct / verified * 100, 1) if verified > 0 else 0
    report.append("-" * 60)
    report.append(f"  Verified: {verified} | Correct: {correct} | Wrong: {wrong} | Flat: {flat} | Skipped: {skipped}")
    if meaningful > 0:
        report.append(f"  Meaningful Accuracy (excl flat): {accuracy}% ({correct}/{meaningful})")
    report.append(f"  Raw Accuracy (incl flat): {raw_accuracy}% ({correct}/{verified})")
    report.append("=" * 60)

    for line in report:
        logger.info(line)

    # Also save to file
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "verify_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info("Report saved to outputs/verify_report.txt")

    # Print daily summary
    summary = get_verification_summary()
    if summary:
        logger.info("")
        logger.info("=== DAILY SUMMARY ===")
        logger.info("  Date: %s", summary.get("date"))
        logger.info("  Total verified: %d | Correct: %d | Wrong: %d | Flat: %d",
                    summary["total"], summary["correct"], summary["wrong"], summary["flat"])
        logger.info("  Accuracy: %s%% | Total P&L (1 lot): %+.0f CNY",
                    summary["accuracy"], summary["total_profit"])
        by_sig = summary.get("by_signal")
        if by_sig is not None and not by_sig.empty:
            for _, row in by_sig.iterrows():
                logger.info("  %s: accuracy %s%% | P&L %+.0f",
                           row["signal"].upper(), row["accuracy"], row["profit"])
        logger.info("======================")

def main():
    parser = argparse.ArgumentParser(description="Futures Alert Monitor")
    parser.add_argument("--now", action="store_true", help="Single scan, then exit")
    parser.add_argument("--query", action="store_true", help="Database query hint")
    parser.add_argument("--verify", action="store_true",
                        help="Verify signal accuracy: compare predicted direction vs actual price change")
    args = parser.parse_args()

    config = load_config()

    if args.query:
        print("Database: outputs/futures_alert.db")
        return

    init_db()

    if args.verify:
        run_verify(config)
        return

    wc = config.get("wechat", {})
    wc_notifier = WeChatNotifier(
        pushplus_token=wc.get("pushplus_token", "") or os.environ.get("PUSHPLUS_TOKEN", ""),
        wecom_webhook=wc.get("wecom_webhook", "") or os.environ.get("WECOM_WEBHOOK", ""),
    )

    notify_cfg = config.get("notify", {})
    dispatcher = PushDispatcher(
        notifier=wc_notifier,
        max_per_minute=notify_cfg.get("max_pushes_per_minute", 10),
        retry_count=notify_cfg.get("retry_count", 1),
    )
    dispatcher.start()

    monitor_cfg = config.get("monitor", {})
    interval_min = int(monitor_cfg.get("check_interval_minutes", 5))
    interval_sec = interval_min * 60
    trading_only = monitor_cfg.get("trading_only", True)

    if not wc.get("pushplus_token"):
        logger.warning("PushPlus token not configured - alerts will not send")

    if args.now:
        logger.info("Running single check...")
        run_check(config, dispatcher)
        logger.info("Single check complete.")
        dispatcher.stop()
        return

    # Long-running monitor
    logger.info("Monitor started | interval=%dmin | trading_only=%s",
                interval_min, trading_only)
    check_count = 0

    while True:
        if trading_only and not is_trading_time():
            time.sleep(60)
            continue

        check_count += 1
        logger.info("Check #%d", check_count)

        try:
            run_check(config, dispatcher)
        except Exception as e:
            logger.error("Check #%d failed: %s", check_count, e, exc_info=True)

        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
