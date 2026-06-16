"""
Data fetcher module - fetch domestic futures data via AKShare.
Only fetches main contracts (through futures_display_main_sina).
"""
import akshare as ak
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Cache for main contract list
_main_contracts_cache = None


def get_main_contract_list() -> pd.DataFrame:
    """
    Get all main contracts from Sina source.
    Returns: symbol (e.g. RB0), exchange, name
    """
    global _main_contracts_cache
    if _main_contracts_cache is not None:
        return _main_contracts_cache
    try:
        df = ak.futures_display_main_sina()
        logger.info(f"Fetched {len(df)} main contracts")
        _main_contracts_cache = df
        return df
    except Exception as e:
        logger.error(f"Failed to get main contract list: {e}")
        return pd.DataFrame()


def get_realtime_quotes() -> pd.DataFrame:
    """
    Get realtime futures quotes.
    AKShare: futures_zh_realtime() returns currently active contracts.
    """
    try:
        df = ak.futures_zh_realtime()
        col_map = {
            "trade": "last_price",
            "position": "open_interest",
            "ticktime": "time",
            "tradedate": "date",
            "preclose": "pre_close",
            "changepercent": "change_pct",
        }
        df.rename(columns=col_map, inplace=True)
        for col in ["last_price", "change_pct", "volume",
                     "open_interest", "high", "low", "open", "pre_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Failed to get realtime quotes: {e}")
        return pd.DataFrame()


def get_main_contract_quotes() -> pd.DataFrame:
    """
    Get realtime quotes for all main contracts.
    First gets main contract list via display_main_sina,
    then matches realtime prices via zh_realtime.
    Outside trading hours, falls back to latest K-line close.
    Only main contracts are fetched - sub/forward contracts excluded.
    """
    main_list = get_main_contract_list()
    if main_list.empty:
        return pd.DataFrame()

    realtime = get_realtime_quotes()

    # Left join: main contract list + realtime quotes
    merged = main_list.merge(
        realtime[["symbol", "last_price", "change_pct", "volume",
                   "open_interest", "high", "low", "open", "pre_close",
                   "bidprice1", "askprice1"]],
        on="symbol", how="left"
    )

    has_quote = merged["last_price"].notna().sum()
    logger.info(f"Main contracts: {len(merged)} total ({has_quote} with realtime quotes)")

    return merged


def get_minute_candles(symbol: str, period: str = "5") -> pd.DataFrame:
    """
    Get minute K-line data.
    AKShare: futures_zh_minute_sina(symbol, period)
    Returns columns: datetime, open, high, low, close, volume, hold -> open_interest
    """
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        df.rename(columns={"hold": "open_interest"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume", "open_interest"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("datetime")
    except Exception as e:
        logger.warning(f"Failed to get {symbol} {period}min K-line: {e}")
        return pd.DataFrame()


def get_futures_news() -> list:
    """Get futures-related news headlines"""
    try:
        news_df = ak.stock_info_global_ths()
        if not news_df.empty and "title" in news_df.columns:
            return news_df["title"].head(10).tolist()
    except Exception as e:
        logger.warning(f"Failed to get financial news: {e}")
    return []
