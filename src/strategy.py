"""
Multi-factor signal analysis with trade suggestions.

Primary trigger: price touches/breaks 20-day band high or low.
Band reversal logic: touch upper band -> SHORT (做空), touch lower band -> LONG (做多).
Multi-factor scoring: daily RSI + daily MACD + daily volume + MA + news.
Hard gate: daily volume resonance + daily indicator resonance required.
Trade engine: entry price, stop loss (ATR-based), band-based targets.
Anti-spam: one alert per symbol per direction per cooldown window.
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
    """Returns 'up' (near upper band, potential short), 'down' (near lower band, potential long), or None.
    Band trading: price approaching/touching boundary triggers signal."""
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
    """0-40: how close price is to the band boundary (band reversal scoring)."""
    if direction == "up":
        penetration = (latest - high) / high if high and high != 0 else 0
    else:
        penetration = (low - latest) / low if low and low != 0 else 0

    # Normalize: 0.5% penetration = full score
    return min(40, max(5, abs(penetration) * 40 / 0.005))


def compute_rsi_score(rsi: Optional[float], direction: str) -> float:
    """0-20: RSI alignment for band reversal (daily 14-period).
    Near upper band (short): overbought RSI scores high (reversal imminent).
    Near lower band (long): oversold RSI scores high."""
    if rsi is None:
        return 0
    if direction == "up":
        # Near upper band -> SHORT. High RSI (overbought) = strong reversal potential.
        if rsi >= 75:       return 20
        elif rsi >= 65:     return 18
        elif rsi >= 60:     return 15
        elif rsi >= 55:     return 10
        elif rsi >= 45:     return 5
        else:               return 0  # not overbought, weak short signal
    else:
        # Near lower band -> LONG. Low RSI (oversold) = strong reversal potential.
        if rsi <= 25:       return 20
        elif rsi <= 35:     return 18
        elif rsi <= 40:     return 15
        elif rsi <= 45:     return 10
        elif rsi <= 55:     return 5
        else:               return 0  # not oversold, weak long signal


def compute_macd_score(macd_bar: Optional[float], macd_bar_prev: Optional[float], direction: str) -> float:
    """0-15: MACD柱收敛 score for band reversal.
    Near upper band (short): MACD柱 positive but shrinking = high score.
    Near lower band (long): MACD柱 negative but shrinking = high score."""
    if macd_bar is None:
        return 0

    # Check柱收敛 (柱 magnitude decreasing, momentum exhausting)
    if macd_bar_prev is not None and not pd.isna(macd_bar_prev):
        converging = abs(macd_bar) < abs(macd_bar_prev)
    else:
        converging = False

    if direction == "up":
        # Near upper band -> SHORT. Ideal: MACD柱 positive but shrinking (momentum fading).
        if macd_bar <= 0:
            return 10  # already turned down, strong
        elif converging:
            return 15  #柱收敛, momentum exhausting
        else:
            return 5   # still expanding, weak
    else:
        # Near lower band -> LONG. Ideal: MACD柱 negative but shrinking.
        if macd_bar >= 0:
            return 10  # already turned up, strong
        elif converging:
            return 15  #柱收敛
        else:
            return 5   # still expanding, weak


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
    """0-10: MA alignment for band reversal.
    Near upper band (short): bearish MA alignment scores high.
    Near lower band (long): bullish MA alignment scores high."""
    if ma5 is None or ma10 is None or ma20 is None:
        return 0
    # Check alignment
    aligned_bull = ma5 > ma10 > ma20
    aligned_bear = ma5 < ma10 < ma20

    if direction == "up":
        # Near upper band -> SHORT. Bearish alignment ideal.
        if aligned_bear:
            return 10
        elif ma5 < ma10:
            return 6
        elif ma5 < ma20:
            return 3
        else:
            return 0
    else:
        # Near lower band -> LONG. Bullish alignment ideal.
        if aligned_bull:
            return 10
        elif ma5 > ma10:
            return 6
        elif ma5 > ma20:
            return 3
        else:
            return 0


def compute_news_score(news_direction: Optional[str], signal_direction: str) -> float:
    """0-5: news alignment bonus.
    Band reversal: bearish news at upper band (short) scores high;
    bullish news at lower band (long) scores high."""
    if news_direction is None:
        return 0
    if signal_direction == "short" and news_direction == "bearish":
        return 5
    if signal_direction == "long" and news_direction == "bullish":
        return 5
    if news_direction == "neutral":
        return 2
    return 0


# ---------------------------------------------------------------------------
# Band reversal confirmation gate (daily volume + daily indicator resonance)
# ---------------------------------------------------------------------------

def confirm_band_reversal(
    rsi: Optional[float],
    macd_bar: Optional[float],
    macd_bar_prev: Optional[float],
    vol_ratio: Optional[float],
    direction: str,
    vol_min: float = 0.9,
) -> bool:
    """Band reversal confirmation: daily volume AND daily indicator must resonate.

    Near upper band (short signal): RSI >= 55 (overbought zone), MACD柱收敛.
    Near lower band (long signal):  RSI <= 45 (oversold zone), MACD柱收敛.
    Volume >= vol_min (daily volume resonance, 量能共振).
    """
    # Volume gate: must have daily volume resonance
    if vol_ratio is None or pd.isna(vol_ratio) or vol_ratio < vol_min:
        return False

    if rsi is None or pd.isna(rsi):
        return False
    if macd_bar is None or pd.isna(macd_bar):
        return False

    if direction == "up":
        # Near upper band -> SHORT. Want overbought RSI and MACD柱收敛 (momentum exhausting).
        rsi_ok = rsi >= 55
        if macd_bar_prev is not None and not pd.isna(macd_bar_prev):
            macd_confirm = abs(macd_bar) < abs(macd_bar_prev)
        else:
            macd_confirm = macd_bar <= 0  # already turning down
    else:
        # Near lower band -> LONG. Want oversold RSI and MACD柱收敛.
        rsi_ok = rsi <= 45
        if macd_bar_prev is not None and not pd.isna(macd_bar_prev):
            macd_confirm = abs(macd_bar) < abs(macd_bar_prev)
        else:
            macd_confirm = macd_bar >= 0  # already turning up

    return rsi_ok and macd_confirm  # BOTH must confirm for band reversal signal


# ---------------------------------------------------------------------------
# Trade suggestion engine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pre-push data integrity validation (Instruction: no erroneous pushes)
# ---------------------------------------------------------------------------

def validate_signal_health(
    symbol: str,
    direction: str,
    range_high: Optional[float],
    range_low: Optional[float],
    latest_price: float,
    volume_ratio: Optional[float],
    rsi: Optional[float],
    macd_bar: Optional[float],
) -> tuple:
    """Validate signal data integrity before push. Returns (passed: bool, reason: str).

    Checks:
      1. Band integrity: range_high > range_low (hard rule)
      2. Volume sanity: volume_ratio <= 5.0 (anomalous spike filter)
      3. Indicator logic: no violent contradiction between RSI and price position
    """
    # --- Check 1: Band integrity ---
    if range_high is None or range_low is None:
        return False, "band data missing"
    if pd.isna(range_high) or pd.isna(range_low):
        return False, "band data NaN"
    if range_high <= range_low:
        logger.warning("%s: REJECTED - range_high(%.2f) <= range_low(%.2f)", symbol, range_high, range_low)
        return False, f"band inverted: high={range_high:.2f} <= low={range_low:.2f}"

    # --- Check 2: Volume anomaly ---
    if volume_ratio is not None and not pd.isna(volume_ratio):
        if volume_ratio > 5.0:
            logger.warning("%s: REJECTED - volume_ratio=%.1f exceeds 5.0x limit", symbol, volume_ratio)
            return False, f"volume spike: {volume_ratio:.1f}x > 5.0x"

    # --- Check 3: Indicator logic consistency ---
    if rsi is not None and not pd.isna(rsi) and latest_price is not None:
        if direction == "up":
            # Near upper band = potential SHORT. RSI should be elevated.
            # RSI < 30 at upper band = likely data anomaly (price high but RSI oversold)
            if rsi < 30:
                logger.warning("%s: REJECTED - SHORT signal but RSI=%.0f (oversold at upper band, data anomaly)", symbol, rsi)
                return False, f"RSI contradiction: {rsi:.0f} (oversold at upper band)"
        else:
            # Near lower band = potential LONG. RSI should be depressed.
            # RSI > 70 at lower band = likely data anomaly (price low but RSI overbought)
            if rsi > 70:
                logger.warning("%s: REJECTED - LONG signal but RSI=%.0f (overbought at lower band, data anomaly)", symbol, rsi)
                return False, f"RSI contradiction: {rsi:.0f} (overbought at lower band)"

    return True, ""


def calculate_trade_suggestions(
    direction: str,
    latest_price: float,
    atr: Optional[float],
    range_high: Optional[float],
    range_low: Optional[float],
    boll_upper: Optional[float],
    boll_lower: Optional[float],
    boll_mid: Optional[float],
    sl_mult: float = 1.5,
    tp_atr_mult: float = 1.0,
) -> Dict:
    """
    Band trading: generate entry, stop-loss, and take-profit levels.

    Direction "up" (near upper band) = SHORT: entry at current, stop above band high, target band low.
    Direction "down" (near lower band) = LONG: entry at current, stop below band low, target band high.
    """
    if atr is None or atr <= 0:
        if range_high and range_low and range_high > range_low:
            atr = (range_high - range_low) * 0.05
        else:
            atr = latest_price * 0.005

    if direction == "up":
        # Near upper band -> SHORT (做空): sell at current
        entry = latest_price
        # Stop: above swing_high (band breakout invalidates the trade)
        sl = round(range_high + sl_mult * atr, 2) if range_high else round(latest_price + 2.0 * atr, 2)
        if range_high:
            sl = round(max(sl, range_high * 1.005), 2)
        # Target 1: swing_low (band lower boundary, mean reversion)
        # Target 2: swing_low - ATR extension (overshoot target)
        tp1 = round(range_low, 2) if range_low else round(latest_price - 2.0 * atr, 2)
        tp2 = round(range_low - tp_atr_mult * atr, 2) if range_low else round(latest_price - 4.0 * atr, 2)
        boll_ref = f"日线布林上轨 {boll_upper:.2f}" if boll_upper else ""
    else:
        # Near lower band -> LONG (做多): buy at current
        entry = latest_price
        # Stop: below swing_low (band breakdown invalidates the trade)
        sl = round(range_low - sl_mult * atr, 2) if range_low else round(latest_price - 2.0 * atr, 2)
        if range_low:
            sl = round(min(sl, range_low * 0.995), 2)
        # Target 1: swing_high (band upper boundary, mean reversion)
        # Target 2: swing_high + ATR extension
        tp1 = round(range_high, 2) if range_high else round(latest_price + 2.0 * atr, 2)
        tp2 = round(range_high + tp_atr_mult * atr, 2) if range_high else round(latest_price + 4.0 * atr, 2)
        boll_ref = f"日线布林下轨 {boll_lower:.2f}" if boll_lower else ""

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
    macd_bar_prev: Optional[float] = None,
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
    all_news: Optional[list] = None,
) -> str:
    """Generate a rich, actionable alert message with trade suggestions.
    News section prioritizes medium/long-term (中期/长期) news for band trading."""

    variety = f"{name} {symbol}"
    emoji = "\U0001f7e2" if direction == "up" else "\U0001f534"
    # Band reversal: up (near high) = short (做空), down (near low) = long (做多)
    dir_cn = "\u505a\u7a7a" if direction == "up" else "\u505a\u591a"

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
        f"\u3010\u6ce2\u6bb5\u5355\u3011{emoji}{dir_cn}",
        f"\u54c1\u79cd\uff1a{variety}",
        f"\u7efc\u5408\u8bc4\u5206\uff1a{score}/100\uff08{score_label}\uff09\uff5c\u6301\u4ed3\u53c2\u8003\uff1a\u6570\u65e5",
        "\u2500\u2500" * 18,
    ]

    # ---- Section 2: Market Data ----
    lines.append("\U0001f4ca \u884c\u60c5\u6570\u636e")
    lines.append(f"  \u5f53\u524d\u4ef7\u683c\uff1a{_fmt(latest_price)}")
    lines.append(f"  \u6ce2\u6bb5\u533a\u95f4\u9ad8\u70b9\uff1a{_fmt(range_high)}  \u6ce2\u6bb5\u533a\u95f4\u4f4e\u70b9\uff1a{_fmt(range_low)}")
    if position_pct is not None:
        lines.append(f"  \u533a\u95f4\u4f4d\u7f6e\uff1a{position_pct:.0f}%")
    if trade.get("boll_ref"):
        lines.append(f"  \u65e5\u7ebf\u5e03\u6797\u53c2\u8003\uff1a{trade['boll_ref']}  \u4e2d\u8f68 {boll_mid:.2f}" if boll_mid else f"  \u65e5\u7ebf\u5e03\u6797\u53c2\u8003\uff1a{trade['boll_ref']}")
    lines.append("\u2500\u2500" * 18)

    # ---- Section 3: Trade Suggestions (波段单，离场参考周期数日) ----
    lines.append("\U0001f3af \u4ea4\u6613\u5efa\u8bae")
    lines.append(f"  \u65b9\u5411\uff1a{dir_cn}")
    lines.append(f"  \u5efa\u8bae\u5165\u573a\uff1a{_fmt(trade['entry'])} \u9644\u8fd1")
    lines.append(f"  \u6b62\u635f\u4f4d\uff1a{_fmt(trade['stop_loss'])}\uff08\u65e5\u7ebfATR 1.5\u500d\uff0c{trade['stop_loss_pct']:+.2f}%\uff09")
    lines.append(f"  \u76ee\u68071\uff1a{_fmt(trade['target_1'])}\uff08\u76c8\u4e8f\u6bd4 {trade['risk_reward_1']}:1\uff09")
    lines.append(f"  \u76ee\u68072\uff1a{_fmt(trade['target_2'])}\uff08\u76c8\u4e8f\u6bd4 {trade['risk_reward_2']}:1\uff09")
    lines.append("\u2500\u2500" * 18)

    # ---- Section 4: Indicator Interpretation ----
    lines.append("\U0001f4c8 \u6307\u6807\u89e3\u8bfb")

    # RSI (14-day, daily-based)
    if rsi is not None and not pd.isna(rsi):
        if rsi >= 65:
            rsi_label = "\u6ce2\u6bb5\u504f\u5f3a"
        elif rsi <= 35:
            rsi_label = "\u6ce2\u6bb5\u504f\u5f31"
        else:
            rsi_label = "\u6ce2\u6bb5\u4e2d\u6027"
        lines.append(f"  RSI(14\u65e5) {rsi:.0f}  \u2192 {rsi_label}")

    # MACD (daily bar comparison: current vs previous)
    if macd_bar is not None and not pd.isna(macd_bar):
        if macd_bar_prev is not None and not pd.isna(macd_bar_prev):
            abs_cur = abs(macd_bar)
            abs_prev = abs(macd_bar_prev)
            if abs_cur > abs_prev:
                change_label = "\u67f1\u653e\u5927"
            elif abs_cur < abs_prev:
                change_label = "\u67f1\u6536\u655b"
            else:
                change_label = "\u67f1\u6301\u5e73"
        else:
            change_label = ""
        if macd_bar > 0:
            macd_label = f"\u7ea2{change_label}" if change_label else "\u7ea2\u67f1"
        else:
            macd_label = f"\u7eff{change_label}" if change_label else "\u7eff\u67f1"
        lines.append(f"  MACD\u65e5\u7ebf\u67f1 {macd_bar:+.2f}  \u2192 {macd_label}")

    # MA alignment (daily MA5/MA10/MA20)
    if ma5 is not None and ma10 is not None and ma20 is not None:
        if ma5 > ma10 > ma20:
            ma_label = "\u6ce2\u6bb5\u591a\u5934\u6392\u5217 (MA5>MA10>MA20)"
        elif ma5 < ma10 < ma20:
            ma_label = "\u6ce2\u6bb5\u7a7a\u5934\u6392\u5217 (MA5<MA10<MA20)"
        elif ma5 > ma20:
            ma_label = "\u77ed\u671f\u504f\u591a"
        elif ma5 < ma20:
            ma_label = "\u77ed\u671f\u504f\u7a7a"
        else:
            ma_label = "\u7c98\u5408\u9707\u8361"
        lines.append(f"  \u65e5\u7ebf\u5747\u7ebf\u6392\u5217\uff1a{ma_label}")

    # Volume (daily-based ratio: latest / 20-day avg)
    if volume_ratio is not None:
        if volume_ratio > 2.0:
            vol_label = f"\u5927\u5e45\u653e\u91cf {volume_ratio:.1f}\u500d"
        elif volume_ratio >= 1.2:
            vol_label = f"\u6e29\u548c\u653e\u91cf {volume_ratio:.1f}\u500d"
        elif volume_ratio >= 0.8:
            vol_label = f"\u5747\u91cf {volume_ratio:.1f}\u500d"
        else:
            vol_label = f"\u7f29\u91cf {volume_ratio:.1f}\u500d"
        lines.append(f"  \u6210\u4ea4\u91cf\uff1a{vol_label}")

    # Open interest change (daily EOD comparison)
    if oi_change_pct is not None:
        if oi_change_pct == 0:
            oi_label = "\u6301\u4ed3\u6301\u5e73"
        elif oi_change_pct > 2:
            oi_label = f"\u6301\u4ed3\u589e\u52a0 {oi_change_pct:+.1f}%\uff08\u8d44\u91d1\u6d41\u5165\uff09"
        elif oi_change_pct > 0:
            oi_label = f"\u6301\u4ed3\u5fae\u589e {oi_change_pct:+.1f}%"
        elif oi_change_pct > -2:
            oi_label = f"\u6301\u4ed3\u5fae\u51cf {oi_change_pct:+.1f}%"
        else:
            oi_label = f"\u6301\u4ed3\u51cf\u5c11 {oi_change_pct:+.1f}%\uff08\u8d44\u91d1\u6d41\u51fa\uff09"
        lines.append(f"  \u6301\u4ed3\u53d8\u5316\uff1a{oi_label}")

    lines.append("\u2500\u2500" * 18)

    # ---- Section 5: News (prioritize band-relevant medium/long-term news) ----
    lines.append("\U0001f4f0 \u76f8\u5173\u8d44\u8baf")

    direction_map = {
        "bullish": "[\u5229\u591a]",
        "bearish": "[\u5229\u7a7a]",
        "neutral": "[\u4e2d\u6027]",
    }

    if all_news and len(all_news) > 0:
        # Separate medium/long-term (band-relevant) from short-term
        band_news = [n for n in all_news if n.get("impact") in ("medium", "long")]
        short_news = [n for n in all_news if n.get("impact") == "short"]

        # Show band-relevant news first (max 3)
        for n in band_news[:3]:
            impact_tag = n.get("impact_label", "\u6ce2\u6bb5")
            tag = direction_map.get(n.get("direction"), "")
            snippet = n["headline"][:80]
            lines.append(f"  [{impact_tag}] {snippet} {tag}")

        # Show short-term news if space (max 2)
        if band_news:
            for n in short_news[:2]:
                impact_tag = n.get("impact_label", "\u77ed\u671f")
                tag = direction_map.get(n.get("direction"), "")
                snippet = n["headline"][:80]
                lines.append(f"  [{impact_tag}] {snippet} {tag}")

        if not band_news:
            # No medium/long-term band-relevant news
            lines.append("  \u6682\u65e0\u6ce2\u6bb5\u76f8\u5173\u8d44\u8baf")
            # Still show a couple short-term items
            for n in short_news[:2]:
                impact_tag = n.get("impact_label", "\u77ed\u671f")
                tag = direction_map.get(n.get("direction"), "")
                snippet = n["headline"][:80]
                lines.append(f"  [{impact_tag}] {snippet} {tag}")
    elif news_summary:
        # Fallback: old-style single news item
        tag = direction_map.get(news_direction, "")
        snippet = news_summary[:100]
        lines.append(f"  [{news_impact_label or '\u77ed\u671f'}] {snippet} {tag}")
    else:
        lines.append("  \u6682\u65e0\u6ce2\u6bb5\u76f8\u5173\u8d44\u8baf")

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
    Band reversal analysis pipeline:
      1. Range boundary detection (primary trigger: price near band edge)
      2. RSI + MACD柱收敛 gating (daily volume + indicator resonance)
      3. Multi-factor scoring (0-100)
      4. Trade suggestion generation (entry / SL / TP, band-based)
      5. Anti-spam cooldown enforcement (per symbol + direction)
      6. Generate rich alert message labeled as 波段单
    """
    if config is None:
        config = {}

    breakout_tolerance = config.get("breakout_tolerance", 0.003)
    cooldown_minutes = config.get("cooldown_minutes", 300)
    min_score = config.get("min_alert_score", 55)  # minimum score to alert

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

        # --- Step 2: Band reversal confirmation gate (daily volume + indicator resonance) ---
        rsi = _safe(_extract_indicator(snap, "rsi"))
        macd_bar = _safe(_extract_indicator(snap, "macd_bar"))
        macd_bar_prev = _safe(snap.get("macd_bar_prev"))
        vol_ratio = _safe(snap.get("volume_ratio"))

        # Hard gate: daily volume resonance + indicator resonance
        if not confirm_band_reversal(rsi, macd_bar, macd_bar_prev, vol_ratio, direction):
            logger.debug(
                "%s: band reversal not confirmed (RSI=%s, MACD=%s, macd_prev=%s, vol=%s)",
                symbol, rsi, macd_bar, macd_bar_prev, vol_ratio,
            )
            continue

        # --- Step 2b: Volume gate (extra check) ---
        vol_ratio_min = config.get("volume_ratio_min", 1.0)
        if vol_ratio is None or vol_ratio < vol_ratio_min:
            logger.debug("%s: rejected - volume_ratio=%.2f below %.1f", symbol, vol_ratio or 0, vol_ratio_min)
            continue

        # --- Step 2c: MA alignment ---
        ma5 = _safe(snap.get("ma5"))
        ma10 = _safe(snap.get("ma10"))

        # --- Step 2c-bis: Trend context filter (price vs MA20) ---
        # Band reversal: short only above MA20 (testing upper band), long only below MA20 (testing lower band)
        if config.get("trend_filter_enabled", True):
            ma20_for_filter = _safe(snap.get("ma20"))
            if ma20_for_filter is not None:
                if direction == "up" and latest_price <= ma20_for_filter:
                    logger.debug("%s: rejected - SHORT but price %.2f <= MA20 %.2f (not at upper band)", symbol, latest_price, ma20_for_filter)
                    continue
                if direction == "down" and latest_price >= ma20_for_filter:
                    logger.debug("%s: rejected - LONG but price %.2f >= MA20 %.2f (not at lower band)", symbol, latest_price, ma20_for_filter)
                    continue

        # --- Step 2c-ter: ATR volatility filter ---
        if config.get("atr_filter_enabled", True):
            atr_for_filter = _safe(snap.get("atr"))
            if atr_for_filter is not None and atr_for_filter > 0:
                atr_pct = atr_for_filter / latest_price
                atr_min = config.get("atr_min_pct", 0.002)
                atr_max = config.get("atr_max_pct", 0.08)
                if atr_pct < atr_min:
                    logger.debug("%s: rejected - ATR/pct=%.4f below min %.4f", symbol, atr_pct, atr_min)
                    continue
                if atr_pct > atr_max:
                    logger.debug("%s: rejected - ATR/pct=%.4f above max %.4f", symbol, atr_pct, atr_max)
                    continue

        # --- Step 3: Multi-factor scoring ---
        # Band reversal: up (near high) = short, down (near low) = long
        signal_type = "short" if direction == "up" else "long"

        score_breakout = compute_breakout_score(latest_price, range_high or 0, range_low or 0, direction)
        score_rsi     = compute_rsi_score(rsi, direction)
        score_macd    = compute_macd_score(macd_bar, macd_bar_prev, direction)

        score_volume  = compute_volume_score(vol_ratio)

        ma20 = _safe(snap.get("ma20"))
        score_ma      = compute_ma_score(ma5, ma10, ma20, direction, latest_price)

        news_dir      = snap.get("news_direction")
        score_news    = compute_news_score(news_dir, signal_type)

        total_score = int(round(score_breakout + score_rsi + score_macd +
                                score_volume + score_ma + score_news))

        # Score threshold
        if signal_type == "long":
            effective_min_score = config.get("bull_min_alert_score", 60)
        else:
            effective_min_score = min_score
        if total_score < effective_min_score:
            logger.debug("%s: score %d below threshold %d", symbol, total_score, effective_min_score)
            continue

        # --- Step 4: Anti-spam (per symbol + signal direction) ---
        spam_key = f"{symbol}:{signal_type}"
        last_ts = anti_spam_cache.get(spam_key, 0)
        if now_ts - last_ts < cooldown_minutes * 60:
            remaining = int(cooldown_minutes - (now_ts - last_ts) / 60)
            logger.debug("%s(%s): cooldown active (%d min remaining)", symbol, signal_type, remaining)
            continue

        # --- Step 4b: Pre-push data integrity validation ---
        health_ok, health_reason = validate_signal_health(
            symbol, direction, range_high, range_low, latest_price,
            vol_ratio, rsi, macd_bar,
        )
        if not health_ok:
            logger.warning("%s: REJECTED by health check - %s", symbol, health_reason)
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
        news_impact_label = snap.get("news_impact_label", "")
        all_news = snap.get("all_news", [])
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
            macd_bar_prev=macd_bar_prev,
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
            news_impact_label=news_impact_label,
            all_news=all_news,
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

        anti_spam_cache[spam_key] = now_ts

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
