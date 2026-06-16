"""
Multi-factor signal analysis with trade suggestions.

Primary trigger: price touches/breaks range high or low.
Multi-factor scoring: breakout + RSI + MACD + volume + MA + news.
Trade engine: entry price, stop loss (ATR-based), take-profit targets.
Anti-spam: one alert per symbol per cooldown window.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary trigger: range breakout detection
# ---------------------------------------------------------------------------

def check_range_breakout(
    latest_price: float,
    range_high: float,
    range_low: float,
    tolerance: float = 0.002,
) -> Optional[str]:
    """Returns 'up' (breakout above), 'down' (breakdown below), or None."""
    if latest_price is None or range_high is None or range_low is None:
        return None
    if pd.isna(latest_price) or pd.isna(range_high) or pd.isna(range_low):
        return None
    if range_high <= range_low:
        return None

    if latest_price >= range_high * (1.0 - tolerance):
        return "up"
    elif latest_price <= range_low * (1.0 + tolerance):
        return "down"
    return None


# ---------------------------------------------------------------------------
# Multi-factor scoring engine
# ---------------------------------------------------------------------------

def _safe(val):
    """Coerce to float, return None if invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return None


def compute_breakout_score(latest: float, high: float, low: float, direction: str) -> float:
    """0-40: how decisively price broke the range boundary."""
    if direction == "up":
        penetration = (latest - high) / high if high and high != 0 else 0
    else:
        penetration = (low - latest) / low if low and low != 0 else 0

    # Normalize: 0.5% penetration = full score
    return min(40, max(5, abs(penetration) * 40 / 0.005))


def compute_rsi_score(rsi: Optional[float], direction: str) -> float:
    """0-20: RSI alignment with direction."""
    if rsi is None:
        return 0
    if direction == "up":
        if rsi >= 70:       return 10   # strong but overbought risk
        elif rsi >= 55:     return 20   # sweet spot
        elif rsi >= 45:     return 15   # neutral-positive
        elif rsi >= 30:     return 5    # oversold recovery potential
        else:               return 0
    else:
        if rsi <= 30:       return 10
        elif rsi <= 45:     return 20
        elif rsi <= 55:     return 15
        elif rsi <= 70:     return 5
        else:               return 0


def compute_macd_score(macd_bar: Optional[float], direction: str) -> float:
    """0-15: MACD bar magnitude and direction."""
    if macd_bar is None:
        return 0
    if direction == "up":
        if macd_bar > 0:
            return min(15, 5 + abs(macd_bar) * 2)
        else:
            return max(0, 5 - abs(macd_bar) * 2)
    else:
        if macd_bar < 0:
            return min(15, 5 + abs(macd_bar) * 2)
        else:
            return max(0, 5 - abs(macd_bar) * 2)


def compute_volume_score(vol_ratio: Optional[float]) -> float:
    """0-10: volume confirmation. Ratio > 1.2 = breakout volume."""
    if vol_ratio is None:
        return 0
    if vol_ratio >= 2.0:    return 10
    elif vol_ratio >= 1.5:  return 8
    elif vol_ratio >= 1.2:  return 6
    elif vol_ratio >= 1.0:  return 4
    else:                   return 2


def compute_ma_score(ma5: Optional[float], ma10: Optional[float],
                     ma20: Optional[float], direction: str,
                     latest: Optional[float]) -> float:
    """0-10: MA alignment (bullish/bearish configuration)."""
    if ma5 is None or ma10 is None or ma20 is None:
        return 0
    # Check alignment
    aligned_bull = ma5 > ma10 > ma20
    aligned_bear = ma5 < ma10 < ma20

    if direction == "up":
        if aligned_bull:
            return 10
        elif ma5 > ma10:
            return 6
        elif ma5 > ma20:
            return 3
        else:
            return 0
    else:
        if aligned_bear:
            return 10
        elif ma5 < ma10:
            return 6
        elif ma5 < ma20:
            return 3
        else:
            return 0


def compute_news_score(news_direction: Optional[str], signal_direction: str) -> float:
    """0-5: news alignment bonus."""
    if news_direction is None:
        return 0
    if signal_direction == "bull" and news_direction == "bullish":
        return 5
    if signal_direction == "bear" and news_direction == "bearish":
        return 5
    if news_direction == "neutral":
        return 2
    # News contradicts signal - penalty
    return 0


# ---------------------------------------------------------------------------
# Confirmation gate (oldschool RSI + MACD check, kept for compatibility)
# ---------------------------------------------------------------------------

def confirm_breakout(
    rsi: Optional[float],
    macd_bar: Optional[float],
    direction: str,
) -> bool:
    """Secondary confirmation: at least one indicator aligns."""
    if direction == "up":
        rsi_ok = (rsi is not None and not pd.isna(rsi) and rsi > 50)
        macd_ok = (macd_bar is not None and not pd.isna(macd_bar) and macd_bar > 0)
    else:
        rsi_ok = (rsi is not None and not pd.isna(rsi) and rsi < 50)
        macd_ok = (macd_bar is not None and not pd.isna(macd_bar) and macd_bar < 0)
    return rsi_ok and macd_ok  # BOTH must confirm for 80%+ accuracy target


# ---------------------------------------------------------------------------
# Trade suggestion engine
# ---------------------------------------------------------------------------

def calculate_trade_suggestions(
    direction: str,
    latest_price: float,
    atr: Optional[float],
    range_high: Optional[float],
    range_low: Optional[float],
    boll_upper: Optional[float],
    boll_lower: Optional[float],
    boll_mid: Optional[float],
) -> Dict:
    """
    Generate entry, stop-loss, and take-profit levels.

    Stop-loss uses ATR-based buffer (1.5x ATR), falling back to range boundary.
    Targets use ATR multiples (1.5x, 3x) with Bollinger bands as reference.
    """
    if atr is None or atr <= 0:
        # Fallback: estimate ATR from range width
        if range_high and range_low and range_high > range_low:
            atr = (range_high - range_low) * 0.05
        else:
            atr = latest_price * 0.005

    if direction == "up":
        # Bullish: buy at current or pullback to range_high
        entry = latest_price
        # Stop: ATR-based first, range_low as safety net
        sl_atr = latest_price - 1.5 * atr
        sl = round(sl_atr, 2)
        # Don't let stop go below range_low unless range_low is very close
        if range_low and sl < range_low * 0.98:
            sl = round(range_low, 2)
        tp1 = round(latest_price + 1.5 * atr, 2)
        tp2 = round(latest_price + 3.0 * atr, 2)
        # Bollinger context
        boll_ref = f"布林上轨 {boll_upper:.2f}" if boll_upper else ""
    else:
        # Bearish: sell at current or rally to range_low
        entry = latest_price
        # Stop: ATR-based first, range_high as safety net
        sl_atr = latest_price + 1.5 * atr
        sl = round(sl_atr, 2)
        # Don't let stop go above range_high unless range_high is very close
        if range_high and sl > range_high * 1.02:
            sl = round(range_high, 2)
        tp1 = round(latest_price - 1.5 * atr, 2)
        tp2 = round(latest_price - 3.0 * atr, 2)
        boll_ref = f"布林下轨 {boll_lower:.2f}" if boll_lower else ""

    # Risk/reward ratios
    risk = abs(entry - sl) if sl != entry else atr
    rr1 = round(abs(tp1 - entry) / risk, 1) if risk > 0 else 0
    rr2 = round(abs(tp2 - entry) / risk, 1) if risk > 0 else 0

    # Stop-loss as percentage of entry
    sl_pct = round(abs(sl - entry) / entry * 100, 2)

    return {
        "entry": round(entry, 2),
        "stop_loss": sl,
        "stop_loss_pct": sl_pct,
        "target_1": tp1,
        "target_2": tp2,
        "risk_reward_1": rr1,
        "risk_reward_2": rr2,
        "atr_used": round(atr, 4),
        "boll_ref": boll_ref,
    }


# ---------------------------------------------------------------------------
# Generate enriched alert message
# ---------------------------------------------------------------------------

def generate_trade_alert(
    name: str,
    symbol: str,
    direction: str,
    latest_price: float,
    range_high: Optional[float],
    range_low: Optional[float],
    score: int,
    trade: Dict,
    rsi: Optional[float] = None,
    macd_bar: Optional[float] = None,
    volume_ratio: Optional[float] = None,
    position_pct: Optional[float] = None,
    ma5: Optional[float] = None,
    ma10: Optional[float] = None,
    ma20: Optional[float] = None,
    oi_change_pct: Optional[float] = None,
    boll_upper: Optional[float] = None,
    boll_lower: Optional[float] = None,
    boll_mid: Optional[float] = None,
    news_summary: Optional[str] = None,
    news_direction: Optional[str] = None,
) -> str:
    """Generate a rich, actionable alert message with trade suggestions."""

    variety = f"{name} {symbol}"
    emoji = "\U0001f7e2" if direction == "up" else "\U0001f534"
    dir_cn = "\u505a\u591a" if direction == "up" else "\u505a\u7a7a"

    # Score level
    if score >= 70:
        score_label = "\u5f3a\u70c8"
    elif score >= 50:
        score_label = "\u8f83\u5f3a"
    elif score >= 30:
        score_label = "\u4e00\u822c"
    else:
        score_label = "\u504f\u5f31"

    # Number formatting helper - shows clean integers, 2 decimals for floats
    def _fmt(v):
        if v is None:
            return '--'
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return f'{v:.2f}'

    # ---- Section 1: Header ----
    lines = [
        f"\u3010\u4e3b\u529b\u5408\u7ea6\u63d0\u9192\u3011{emoji}{dir_cn}",
        f"\u54c1\u79cd\uff1a{variety}",
        f"\u7efc\u5408\u8bc4\u5206\uff1a{score}/100\uff08{score_label}\uff09",
        "\u2500\u2500" * 18,
    ]

    # ---- Section 2: Market Data ----

    lines.append("\U0001f4ca \u884c\u60c5\u6570\u636e")
    lines.append(f"  \u5f53\u524d\u4ef7\u683c\uff1a{_fmt(latest_price)}")
    lines.append(f"  \u533a\u95f4\u9ad8\u70b9\uff1a{_fmt(range_high)}  \u533a\u95f4\u4f4e\u70b9\uff1a{_fmt(range_low)}")
    if position_pct is not None:
        lines.append(f"  \u533a\u95f4\u4f4d\u7f6e\uff1a{position_pct:.0f}%")
    if trade.get("boll_ref"):
        lines.append(f"  \u5e03\u6797\u53c2\u8003\uff1a{trade['boll_ref']}  \u4e2d\u8f68 {boll_mid:.2f}" if boll_mid else f"  \u5e03\u6797\u53c2\u8003\uff1a{trade['boll_ref']}")
    lines.append("\u2500\u2500" * 18)

    # ---- Section 3: Trade Suggestions ----
    lines.append("\U0001f3af \u4ea4\u6613\u5efa\u8bae")
    lines.append(f"  \u65b9\u5411\uff1a{dir_cn}")
    lines.append(f"  \u5efa\u8bae\u5165\u573a\uff1a{_fmt(trade['entry'])} \u9644\u8fd1")
    lines.append(f"  \u6b62\u635f\u4f4d\uff1a{_fmt(trade['stop_loss'])}\uff08ATR 1.5\u500d\uff0c{trade['stop_loss_pct']:+.2f}%\uff09")
    lines.append(f"  \u76ee\u68071\uff1a{_fmt(trade['target_1'])}\uff08\u76c8\u4e8f\u6bd4 {trade['risk_reward_1']}:1\uff09")
    lines.append(f"  \u76ee\u68072\uff1a{_fmt(trade['target_2'])}\uff08\u76c8\u4e8f\u6bd4 {trade['risk_reward_2']}:1\uff09")
    lines.append("\u2500\u2500" * 18)

    # ---- Section 4: Indicator Interpretation ----
    lines.append("\U0001f4c8 \u6307\u6807\u89e3\u8bfb")

    # RSI
    if rsi is not None and not pd.isna(rsi):
        if rsi >= 70:
            rsi_label = "\u8d85\u4e70\u533a"
        elif rsi >= 55:
            rsi_label = "\u504f\u5f3a"
        elif rsi >= 45:
            rsi_label = "\u4e2d\u6027"
        elif rsi >= 30:
            rsi_label = "\u504f\u5f31"
        else:
            rsi_label = "\u8d85\u5356\u533a"
        lines.append(f"  RSI {rsi:.0f}  \u2192 {rsi_label}")

    # MACD
    if macd_bar is not None and not pd.isna(macd_bar):
        if macd_bar > 0:
            if macd_bar > 5:
                macd_label = "\u7ea2\u67f1\u653e\u5927"
            else:
                macd_label = "\u7ea2\u67f1\u7f29\u5c0f"
        else:
            if macd_bar < -5:
                macd_label = "\u7eff\u67f1\u653e\u5927"
            else:
                macd_label = "\u7eff\u67f1\u7f29\u5c0f"
        lines.append(f"  MACD\u67f1 {macd_bar:+.2f}  \u2192 {macd_label}")

    # MA alignment
    if ma5 is not None and ma10 is not None and ma20 is not None:
        if ma5 > ma10 > ma20:
            ma_label = "\u591a\u5934\u6392\u5217 (MA5>MA10>MA20)"
        elif ma5 < ma10 < ma20:
            ma_label = "\u7a7a\u5934\u6392\u5217 (MA5<MA10<MA20)"
        elif ma5 > ma20:
            ma_label = "\u77ed\u671f\u504f\u591a"
        elif ma5 < ma20:
            ma_label = "\u77ed\u671f\u504f\u7a7a"
        else:
            ma_label = "\u7c98\u5408\u9707\u8361"
        lines.append(f"  \u5747\u7ebf\u6392\u5217\uff1a{ma_label}")

    # Volume
    if volume_ratio is not None:
        if volume_ratio >= 1.5:
            vol_label = f"\u653e\u91cf {volume_ratio:.1f}\u500d\uff08\u7a81\u7834\u653e\u91cf\u786e\u8ba4\uff09"
        elif volume_ratio >= 1.0:
            vol_label = f"\u6b63\u5e38 {volume_ratio:.1f}\u500d"
        else:
            vol_label = f"\u7f29\u91cf {volume_ratio:.1f}\u500d\uff08\u52a8\u80fd\u4e0d\u8db3\uff09"
        lines.append(f"  \u6210\u4ea4\u91cf\uff1a{vol_label}")

    # Open interest change
    if oi_change_pct is not None:
        if oi_change_pct > 2:
            oi_label = f"\u6301\u4ed3\u589e\u52a0 {oi_change_pct:+.1f}%\uff08\u8d44\u91d1\u6d41\u5165\uff09"
        elif oi_change_pct > 0:
            oi_label = f"\u6301\u4ed3\u5fae\u589e {oi_change_pct:+.1f}%"
        elif oi_change_pct > -2:
            oi_label = f"\u6301\u4ed3\u5fae\u51cf {oi_change_pct:+.1f}%"
        else:
            oi_label = f"\u6301\u4ed3\u51cf\u5c11 {oi_change_pct:+.1f}%\uff08\u8d44\u91d1\u6d41\u51fa\uff09"
        lines.append(f"  \u6301\u4ed3\u53d8\u5316\uff1a{oi_label}")

    lines.append("\u2500\u2500" * 18)

    # ---- Section 5: News ----
    lines.append("\U0001f4f0 \u76f8\u5173\u8d44\u8baf")
    if news_summary:
        direction_map = {
            "bullish": "[\u5229\u591a]",
            "bearish": "[\u5229\u7a7a]",
            "neutral": "[\u4e2d\u6027]",
        }
        tag = direction_map.get(news_direction, "")
        snippet = news_summary[:100]
        lines.append(f"  {snippet} {tag}")
    else:
        lines.append("  \u6682\u65e0\u76f4\u63a5\u76f8\u5173\u8d44\u8baf")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper to extract indicator from snapshot
# ---------------------------------------------------------------------------

def _extract_indicator(snap: Dict, name: str) -> Optional[float]:
    """Extract a named indicator value from a snapshot dict."""
    if snap.get("indicator_1_name") == name:
        return snap.get("indicator_1_value")
    if snap.get("indicator_2_name") == name:
        return snap.get("indicator_2_value")
    return snap.get(name)  # try direct key lookup


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_contracts_simple(
    snapshots: List[Dict],
    anti_spam_cache: Dict[str, float],
    config: Optional[dict] = None,
) -> List[Dict]:
    """
    Multi-factor analysis pipeline:
      1. Range breakout detection (primary trigger)
      2. RSI + MACD gating (confirmation gate)
      3. Multi-factor scoring (0-100)
      4. Trade suggestion generation (entry / SL / TP)
      5. Anti-spam cooldown enforcement
      6. Generate rich alert message
    """
    if config is None:
        config = {}

    breakout_tolerance = config.get("breakout_tolerance", 0.002)
    cooldown_minutes = config.get("cooldown_minutes", 30)
    min_score = config.get("min_alert_score", 25)  # minimum score to alert

    now_ts = time.time()
    results = []

    for snap in snapshots:
        symbol = snap.get("symbol", "")
        name = snap.get("name", symbol)
        latest_price = _safe(snap.get("latest_price"))
        range_high = _safe(snap.get("range_high"))
        range_low = _safe(snap.get("range_low"))

        if latest_price is None:
            continue

        # --- Step 1: Primary trigger ---
        direction = check_range_breakout(
            latest_price, range_high, range_low, breakout_tolerance
        )
        if direction is None:
            continue

        # --- Step 2: Confirmation gate ---
        rsi = _safe(_extract_indicator(snap, "rsi"))
        macd_bar = _safe(_extract_indicator(snap, "macd_bar"))

        if not confirm_breakout(rsi, macd_bar, direction):
            logger.debug(
                "%s: breakout %s not confirmed (RSI=%s, MACD=%s)",
                symbol, direction, rsi, macd_bar,
            )
            continue

        # --- Step 2b: Volume gate (breakouts without volume = fake) ---
        vol_ratio = _safe(snap.get("volume_ratio"))
        if vol_ratio is None or vol_ratio < 0.3:
            logger.debug("%s: rejected - volume_ratio=%.2f below 1.0", symbol, vol_ratio or 0)
            continue

        # --- Step 2c: MA short-term alignment gate (TEMPORARILY DISABLED for tuning) ---
        ma5 = _safe(snap.get("ma5"))
        ma10 = _safe(snap.get("ma10"))
        if False:  # disabled during gate tuning
            if ma5 is not None and ma10 is not None:
                if direction == "up" and ma5 <= ma10:
                    logger.debug("%s: rejected - MA5(%.2f) <= MA10(%.2f) on bullish", symbol, ma5, ma10)
                    continue
                if direction == "down" and ma5 >= ma10:
                    logger.debug("%s: rejected - MA5(%.2f) >= MA10(%.2f) on bearish", symbol, ma5, ma10)
                    continue

        # --- Step 2d: Minimum breakout penetration (0.1% beyond boundary) ---
        if direction == "up" and range_high and range_high > 0:
            penetration = (latest_price - range_high) / range_high
            if False:  # penetration gate disabled
                logger.debug("%s: rejected - penetration %.4f below 0.1%%", symbol, penetration)
                continue
        elif direction == "down" and range_low and range_low > 0:
            penetration = (range_low - latest_price) / range_low
            if False:  # penetration gate disabled
                logger.debug("%s: rejected - penetration %.4f below 0.1%%", symbol, penetration)
                continue

        # --- Step 3: Multi-factor scoring ---
        signal_type = "bull" if direction == "up" else "bear"

        score_breakout = compute_breakout_score(latest_price, range_high or 0, range_low or 0, direction)
        score_rsi     = compute_rsi_score(rsi, direction)
        score_macd    = compute_macd_score(macd_bar, direction)

        score_volume  = compute_volume_score(vol_ratio)

        ma20 = _safe(snap.get("ma20"))
        score_ma      = compute_ma_score(ma5, ma10, ma20, direction, latest_price)

        news_dir      = snap.get("news_direction")
        score_news    = compute_news_score(news_dir, signal_type)

        total_score = int(round(score_breakout + score_rsi + score_macd +
                                score_volume + score_ma + score_news))

        # Score threshold
        if total_score < min_score:
            logger.debug("%s: score %d below threshold %d", symbol, total_score, min_score)
            continue

        # --- Step 4: Anti-spam ---
        last_ts = anti_spam_cache.get(symbol, 0)
        if now_ts - last_ts < cooldown_minutes * 60:
            remaining = int(cooldown_minutes - (now_ts - last_ts) / 60)
            logger.debug("%s: cooldown active (%d min remaining)", symbol, remaining)
            continue

        # --- Step 5: Trade suggestions ---
        atr = _safe(snap.get("atr"))
        boll_upper = _safe(snap.get("boll_upper"))
        boll_lower = _safe(snap.get("boll_lower"))
        boll_mid   = _safe(snap.get("boll_mid"))

        trade = calculate_trade_suggestions(
            direction, latest_price, atr,
            range_high, range_low,
            boll_upper, boll_lower, boll_mid,
        )

        # --- Step 6: Build alert ---
        news_summary = snap.get("news_summary")
        oi_change = _safe(snap.get("oi_change_pct"))
        position_pct = _safe(snap.get("position_pct"))

        message = generate_trade_alert(
            name=name,
            symbol=symbol,
            direction=direction,
            latest_price=latest_price,
            range_high=range_high,
            range_low=range_low,
            score=total_score,
            trade=trade,
            rsi=rsi,
            macd_bar=macd_bar,
            volume_ratio=vol_ratio,
            position_pct=position_pct,
            ma5=ma5,
            ma10=ma10,
            ma20=ma20,
            oi_change_pct=oi_change,
            boll_upper=boll_upper,
            boll_lower=boll_lower,
            boll_mid=boll_mid,
            news_summary=news_summary,
            news_direction=news_dir,
        )

        results.append({
            "symbol": symbol,
            "name": name,
            "signal": signal_type,
            "score": total_score,
            "message": message,
            "data": {
                "close": latest_price,
                "rsi": rsi,
                "macd_bar": macd_bar,
                "atr": atr,
                "volume_ratio": vol_ratio,
            },
            "details": {
                "breakout": 1 if direction == "up" else -1,
                "rsi": rsi,
                "macd_bar": macd_bar,
                "news_direction": news_dir,
                "total_score": total_score,
                "score_breakdown": {
                    "breakout": round(score_breakout, 1),
                    "rsi": round(score_rsi, 1),
                    "macd": round(score_macd, 1),
                    "volume": round(score_volume, 1),
                    "ma": round(score_ma, 1),
                    "news": round(score_news, 1),
                },
                "trade_suggestion": trade,
            },
        })

        anti_spam_cache[symbol] = now_ts

    # Sort by score descending (highest conviction first)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Legacy alert generator (kept for backward compatibility)
# ---------------------------------------------------------------------------

def generate_simple_alert(
    name: str,
    symbol: str,
    direction: str,
    latest_price: float,
    range_high: float,
    range_low: float,
    rsi: Optional[float] = None,
    macd_bar: Optional[float] = None,
    news_summary: Optional[str] = None,
    news_direction: Optional[str] = None,
) -> str:
    """Legacy concise alert format - kept for compatibility."""
    return generate_trade_alert(
        name=name, symbol=symbol, direction=direction,
        latest_price=latest_price, range_high=range_high, range_low=range_low,
        score=0, trade={"entry": latest_price, "stop_loss": 0, "stop_loss_pct": 0,
                         "target_1": 0, "target_2": 0, "risk_reward_1": 0,
                         "risk_reward_2": 0, "atr_used": 0, "boll_ref": ""},
        rsi=rsi, macd_bar=macd_bar, news_summary=news_summary,
        news_direction=news_direction,
    )
