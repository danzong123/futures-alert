"""
技术指标计算模块
"""
import numpy as np
import pandas as pd


def calc_ma(series: pd.Series, period: int) -> pd.Series:
    """移动平均线"""
    return series.rolling(window=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (相对强弱指标)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指标"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def calc_bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """布林带"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (平均真实波幅)"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全部技术指标, 返回带指标的DataFrame
    要求输入至少有: close, high, low, volume, open_interest
    """
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]
    oi = result.get("open_interest", pd.Series(index=df.index, dtype=float))

    # MA
    result["ma5"] = calc_ma(close, 5)
    result["ma10"] = calc_ma(close, 10)
    result["ma20"] = calc_ma(close, 20)
    result["ma60"] = calc_ma(close, 60)

    # MACD
    result["macd_dif"], result["macd_dea"], result["macd_bar"] = calc_macd(close)

    # RSI
    result["rsi"] = calc_rsi(close, 14)

    # BOLL
    result["boll_upper"], result["boll_mid"], result["boll_lower"] = calc_bollinger(close)

    # ATR
    result["atr"] = calc_atr(high, low, close, 14)

    # 成交量均线
    result["volume_ma5"] = calc_ma(volume, 5)
    result["volume_ratio"] = volume / result["volume_ma5"].replace(0, np.nan)

    # 持仓量变化率 (当日 vs 前一根)
    result["oi_change_pct"] = oi.pct_change() * 100 if oi.notna().any() else 0

    # 高点/低点突破检测
    result["hhv_20"] = high.rolling(window=20).max()
    result["llv_20"] = low.rolling(window=20).min()
    result["hhv_10"] = high.rolling(window=10).max()
    result["llv_10"] = low.rolling(window=10).min()

    # 价格位置: 当前价在20日区间的位置百分比
    range_20 = result["hhv_20"] - result["llv_20"]
    result["position_pct"] = ((close - result["llv_20"]) / range_20.replace(0, np.nan)) * 100

    return result
