"""
Database module - SQLite storage for signals, alerts, trades, and core contract data.
"""
import sqlite3
import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Primary data store - uses QH workspace to avoid file lock issues
DB_DIR = os.path.join(os.environ.get(
    "TRADE_DB_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
))
os.makedirs(DB_DIR, exist_ok=True)

# Contract multipliers (CNY per point for 1 lot)
CONTRACT_MULTIPLIERS = {
    "RB": 10, "HC": 10, "I": 100, "J": 100, "JM": 60, "SS": 5,
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AO": 20,
    "AU": 1000, "AG": 15,
    "SC": 1000, "FU": 10, "LU": 10, "BU": 10, "NR": 10, "RU": 10,
    "MA": 10, "TA": 5, "EG": 10, "PF": 5, "PR": 5, "PX": 5,
    "PL": 5, "PP": 5, "L": 5, "V": 5, "EB": 5,
    "SA": 20, "SH": 20, "UR": 20, "FG": 20,
    "M": 10, "Y": 10, "P": 10, "OI": 10, "RM": 10,
    "C": 10, "CS": 10, "A": 10, "B": 10, "CF": 5, "SR": 10,
    "AP": 10, "CJ": 5, "JD": 10, "LH": 16, "PG": 20,
    "SI": 5, "LC": 1, "EC": 50,
    "IF": 300, "IC": 200, "IH": 300, "IM": 200,
    "BC": 5, "PK": 5, "CY": 5, "JR": 20, "RI": 20, "LR": 20,
    "WH": 20, "SM": 5, "SF": 5, "RS": 10, "SP": 10,
    "WR": 10, "AD": 5, "PD": 10,
}

DB_PATH = os.path.join(DB_DIR, "futures_alert.db")


def get_connection():
    """Get database connection"""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_timestamp():
    """Standard timestamp format: YYYYMMDD HH:MM:SS"""
    return datetime.now().strftime("%Y%m%d %H:%M:%S")


def init_db():
    """Initialize database tables"""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                signal TEXT NOT NULL,
                score INTEGER NOT NULL,
                price REAL,
                change_pct REAL,
                details TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                channel TEXT DEFAULT 'pushplus',
                status TEXT DEFAULT 'sent',
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('long','short')),
                signal_id INTEGER,
                signal_type TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                quantity INTEGER DEFAULT 1,
                profit REAL,
                profit_pct REAL,
                status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed')),
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (signal_id) REFERENCES signal_log(id)
            );

            -- Core data table: main contract snapshot
            CREATE TABLE IF NOT EXISTS main_contract_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT '00000000 00:00:00',
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                range_high REAL,
                range_low REAL,
                latest_price REAL,
                indicator_1_name TEXT,
                indicator_1_value REAL,
                indicator_2_name TEXT,
                indicator_2_value REAL,
                news_summary TEXT,
                news_direction TEXT CHECK(news_direction IN ('bullish','bearish','neutral') OR news_direction IS NULL),
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_log(symbol);
            CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_mcd_symbol ON main_contract_data(symbol);
            CREATE INDEX IF NOT EXISTS idx_mcd_timestamp ON main_contract_data(timestamp);

            -- Daily verification log: track signal accuracy vs actual price movement
            CREATE TABLE IF NOT EXISTS verification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                verify_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                signal TEXT NOT NULL,
                score INTEGER,
                signal_price REAL NOT NULL,
                verify_price REAL NOT NULL,
                change_pct REAL NOT NULL,
                is_correct INTEGER NOT NULL DEFAULT 0,
                is_flat INTEGER NOT NULL DEFAULT 0,
                contract_mult REAL DEFAULT 10,
                profit_per_lot REAL DEFAULT 0,
                entry_price REAL,
                stop_loss REAL,
                target_1 REAL,
                target_2 REAL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (signal_id) REFERENCES signal_log(id)
            );

            CREATE INDEX IF NOT EXISTS idx_verify_date ON verification_log(verify_date);
            CREATE INDEX IF NOT EXISTS idx_verify_signal ON verification_log(signal);
        """)
        conn.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
    finally:
        conn.close()


def save_signals(signals: List[Dict]):
    """Save signal records"""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        for s in signals:
            conn.execute(
                """INSERT INTO signal_log (timestamp, symbol, name, signal, score, price, change_pct, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    s.get("symbol", ""),
                    s.get("name", ""),
                    s.get("signal", ""),
                    s.get("score", 0),
                    s.get("data", {}).get("close", 0),
                    s.get("data", {}).get("change_pct", 0),
                    json.dumps(s.get("details", {}), ensure_ascii=False),
                )
            )
        conn.commit()
        logger.info(f"Saved {len(signals)} signal(s)")
    except Exception as e:
        logger.error(f"Save signals failed: {e}")
    finally:
        conn.close()


def save_alert(title: str, content: str, channel: str = "pushplus"):
    """Save alert record"""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn.execute(
            "INSERT INTO alert_history (timestamp, title, content, channel) VALUES (?, ?, ?, ?)",
            (now, title, content, channel)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Save alert failed: {e}")
    finally:
        conn.close()


def get_signal_history(hours: int = 24) -> pd.DataFrame:
    """Get signal history for the last N hours"""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT * FROM signal_log
               WHERE timestamp >= datetime('now', 'localtime', ?)
               ORDER BY timestamp DESC""",
            conn,
            params=(f"-{hours} hours",),
        )
        return df
    except Exception as e:
        logger.error(f"Signal query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_alert_history(hours: int = 24) -> pd.DataFrame:
    """Get alert history for the last N hours"""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT * FROM alert_history
               WHERE timestamp >= datetime('now', 'localtime', ?)
               ORDER BY timestamp DESC""",
            conn,
            params=(f"-{hours} hours",),
        )
        return df
    except Exception as e:
        logger.error(f"Alert query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def save_daily_snapshot(date: str, snapshot: dict):
    """Save daily snapshot"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO daily_snapshot (date, snapshot_json) VALUES (?, ?)",
            (date, json.dumps(snapshot, ensure_ascii=False)),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Save snapshot failed: {e}")
    finally:
        conn.close()


# ============================================================
# CORE DATA TABLE FUNCTIONS
# ============================================================

def save_contract_snapshot(
    symbol: str,
    name: str,
    range_high: float,
    range_low: float,
    latest_price: float,
    indicator_1_name: str = None,
    indicator_1_value: float = None,
    indicator_2_name: str = None,
    indicator_2_value: float = None,
    news_summary: str = None,
    news_direction: str = None,
):
    """Save a single contract snapshot row to main_contract_data."""
    conn = get_connection()
    ts = get_timestamp()
    try:
        conn.execute(
            """INSERT INTO main_contract_data
               (timestamp, symbol, name, range_high, range_low, latest_price,
                indicator_1_name, indicator_1_value,
                indicator_2_name, indicator_2_value,
                news_summary, news_direction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, symbol, name, range_high, range_low, latest_price,
             indicator_1_name, indicator_1_value,
             indicator_2_name, indicator_2_value,
             news_summary, news_direction),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Save contract snapshot failed for {symbol}: {e}")
    finally:
        conn.close()


def save_contract_snapshots_batch(records: List[Dict]):
    """Batch save contract snapshots."""
    if not records:
        return
    conn = get_connection()
    ts = get_timestamp()
    try:
        rows = []
        for r in records:
            rows.append((
                ts,
                r.get("symbol", ""),
                r.get("name", ""),
                r.get("range_high"),
                r.get("range_low"),
                r.get("latest_price"),
                r.get("indicator_1_name"),
                r.get("indicator_1_value"),
                r.get("indicator_2_name"),
                r.get("indicator_2_value"),
                r.get("news_summary"),
                r.get("news_direction"),
            ))
        conn.executemany(
            """INSERT INTO main_contract_data
               (timestamp, symbol, name, range_high, range_low, latest_price,
                indicator_1_name, indicator_1_value,
                indicator_2_name, indicator_2_value,
                news_summary, news_direction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        logger.info(f"Saved {len(records)} contract snapshot(s)")
    except Exception as e:
        logger.error(f"Batch save snapshots failed: {e}")
    finally:
        conn.close()


def get_latest_snapshots() -> pd.DataFrame:
    """Get the most recent snapshot for each contract."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT m.* FROM main_contract_data m
               INNER JOIN (
                   SELECT symbol, MAX(timestamp) AS max_ts
                   FROM main_contract_data
                   GROUP BY symbol
               ) latest ON m.symbol = latest.symbol AND m.timestamp = latest.max_ts
               ORDER BY m.symbol""",
            conn,
        )
        return df
    except Exception as e:
        logger.error(f"Query latest snapshots failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_contract_history(symbol: str, hours: int = 24) -> pd.DataFrame:
    """Get snapshot history for a single contract."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT * FROM main_contract_data
               WHERE symbol = ?
                 AND timestamp >= datetime('now', 'localtime', ?)
               ORDER BY timestamp DESC""",
            conn,
            params=(symbol, f"-{hours} hours"),
        )
        return df
    except Exception as e:
        logger.error(f"Query contract history failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# ============================================================
# TRADE TRACKING FUNCTIONS
# ============================================================

def add_trade(symbol: str, name: str, direction: str, entry_price: float,
              quantity: int = 1, signal_id: int = None, notes: str = ""):
    """Record a new trade"""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur = conn.execute(
            """INSERT INTO trades (symbol, name, direction, entry_price, entry_date,
               quantity, signal_id, notes, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (symbol.upper(), name, direction, entry_price, now, quantity, signal_id, notes)
        )
        conn.commit()
        trade_id = cur.lastrowid
        logger.info(f"Trade #{trade_id} opened: {symbol} {direction} @ {entry_price}")
        return trade_id
    except Exception as e:
        logger.error(f"Add trade failed: {e}")
        return None
    finally:
        conn.close()


def close_trade(trade_id: int, exit_price: float, notes: str = ""):
    """Close an existing trade and calculate profit"""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        row = conn.execute(
            "SELECT id, symbol, direction, entry_price, quantity, notes FROM trades WHERE id = ? AND status = 'open'",
            (trade_id,)
        ).fetchone()
        if not row:
            logger.warning(f"Trade #{trade_id} not found or already closed")
            return False

        entry_price = row["entry_price"]
        direction = row["direction"]
        quantity = row["quantity"]

        if direction == "long":
            profit = (exit_price - entry_price) * quantity
            profit_pct = (exit_price - entry_price) / entry_price * 100
        else:
            profit = (entry_price - exit_price) * quantity
            profit_pct = (entry_price - exit_price) / entry_price * 100

        old_notes = row["notes"] or ""
        if notes:
            old_notes = (old_notes + " | " + notes) if old_notes else notes

        conn.execute(
            """UPDATE trades SET exit_price=?, exit_date=?, profit=?, profit_pct=?,
               status='closed', notes=?, updated_at=? WHERE id=?""",
            (exit_price, now, round(profit, 2), round(profit_pct, 2),
             old_notes, now, trade_id)
        )
        conn.commit()
        logger.info(f"Trade #{trade_id} closed: {row['symbol']} PnL={profit:.2f} ({profit_pct:+.2f}%)")
        return True
    except Exception as e:
        logger.error(f"Close trade failed: {e}")
        return False
    finally:
        conn.close()


def get_all_trades(status: str = None) -> pd.DataFrame:
    """Get all trades, optionally filtered by status"""
    conn = get_connection()
    try:
        if status:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE status = ? ORDER BY entry_date DESC",
                conn, params=(status,)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY entry_date DESC", conn
            )
        return df
    except Exception as e:
        logger.error(f"Trades query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_performance_summary() -> Dict:
    """Get aggregate performance statistics"""
    conn = get_connection()
    try:
        closed = conn.execute(
            "SELECT COUNT(*) as total, ROUND(SUM(profit), 2) as total_pnl, "
            "ROUND(AVG(profit_pct), 2) as avg_pct, "
            "SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as losses "
            "FROM trades WHERE status = 'closed'"
        ).fetchone()

        open_count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'open'"
        ).fetchone()[0]

        best = conn.execute(
            "SELECT symbol, name, direction, profit, profit_pct FROM trades "
            "WHERE status = 'closed' ORDER BY profit DESC LIMIT 1"
        ).fetchone()

        worst = conn.execute(
            "SELECT symbol, name, direction, profit, profit_pct FROM trades "
            "WHERE status = 'closed' ORDER BY profit ASC LIMIT 1"
        ).fetchone()

        by_symbol = pd.read_sql_query(
            """SELECT symbol, name, COUNT(*) as trades,
               SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as losses,
               ROUND(SUM(profit), 2) as total_pnl,
               ROUND(AVG(profit_pct), 2) as avg_pct
               FROM trades WHERE status = 'closed'
               GROUP BY symbol ORDER BY total_pnl DESC""",
            conn
        )

        total = closed["total"] or 0
        wins = closed["wins"] or 0
        losses = closed["losses"] or 0
        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "total_trades": total,
            "open_trades": open_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": closed["total_pnl"] or 0,
            "avg_profit_pct": closed["avg_pct"] or 0,
            "best_trade": dict(best) if best else None,
            "worst_trade": dict(worst) if worst else None,
            "by_symbol": by_symbol,
        }
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        return {}
    finally:
        conn.close()


def delete_trade(trade_id: int):
    """Delete a trade record"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        conn.commit()
        logger.info(f"Trade #{trade_id} deleted")
        return True
    except Exception as e:
        logger.error(f"Delete trade failed: {e}")
        return False
    finally:
        conn.close()


def update_trade(trade_id: int, **kwargs):
    """Update trade fields"""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    allowed = ["entry_price", "exit_price", "direction", "quantity", "notes", "symbol", "name"]
    sets = []
    params = []
    for k, v in kwargs.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return False
    sets.append("updated_at = ?")
    params.append(now)
    params.append(trade_id)
    try:
        conn.execute(
            f"UPDATE trades SET {', '.join(sets)} WHERE id = ?", params
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Update trade failed: {e}")
        return False
    finally:
        conn.close()

# ============================================================
# SIGNAL VERIFICATION & ACCURACY ANALYSIS
# ============================================================

def get_signal_trade_analysis(hours: int = 168) -> pd.DataFrame:
    """Join signals with trades to evaluate signal accuracy.
    Returns DataFrame with signal-trade pairs for directional analysis."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT
                s.id AS signal_id,
                s.timestamp AS signal_time,
                s.symbol,
                s.name,
                s.signal,
                s.score,
                s.price AS signal_price,
                t.id AS trade_id,
                t.direction AS trade_direction,
                t.entry_price,
                t.exit_price,
                t.profit,
                t.profit_pct,
                t.status AS trade_status,
                t.entry_date,
                t.exit_date
            FROM signal_log s
            LEFT JOIN trades t ON s.id = t.signal_id
            WHERE s.timestamp >= datetime('now', 'localtime', ?)
            ORDER BY s.timestamp DESC""",
            conn,
            params=(f"-{hours} hours",),
        )
        if not df.empty:
            def _aligned(row):
                if pd.isna(row['trade_direction']): return None
                return (row['signal'] == 'bull' and row['trade_direction'] == 'long') or \
                       (row['signal'] == 'bear' and row['trade_direction'] == 'short')
            df['direction_aligned'] = df.apply(_aligned, axis=1)

            def _accurate(row):
                if pd.isna(row['profit']): return None
                if row['direction_aligned'] is not True: return False
                return row['profit'] > 0
            df['is_accurate'] = df.apply(_accurate, axis=1)

        return df
    except Exception as e:
        logger.error(f"Signal-trade analysis failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_signal_accuracy_summary(hours: int = 168) -> dict:
    """Summary stats for signal accuracy dashboard."""
    df = get_signal_trade_analysis(hours)
    if df.empty:
        return {}

    total = len(df)
    traded = df['trade_id'].notna().sum()
    closed = (df['trade_status'] == 'closed').sum()
    accurate = (df['is_accurate'] == True).sum()
    inaccurate = (df['is_accurate'] == False).sum()

    # By signal type
    by_signal = df.groupby('signal').agg(
        total=('signal_id', 'count'),
        traded=('trade_id', lambda x: x.notna().sum()),
        accurate=('is_accurate', lambda x: (x == True).sum()),
        inaccurate=('is_accurate', lambda x: (x == False).sum()),
        avg_profit=('profit', 'mean'),
        total_profit=('profit', 'sum'),
    ).reset_index()

    if not by_signal.empty:
        by_signal['accuracy_pct'] = by_signal.apply(
            lambda r: round(r['accurate'] / (r['accurate'] + r['inaccurate']) * 100, 1)
            if (r['accurate'] + r['inaccurate']) > 0 else 0, axis=1
        )

    # By score range
    closed_df = df[df['trade_status'] == 'closed'].copy()
    score_buckets = pd.DataFrame()
    if not closed_df.empty:
        bins = [0, 30, 45, 55, 70, 100]
        labels = ['0-30', '30-45', '45-55', '55-70', '70-100']
        closed_df['score_range'] = pd.cut(closed_df['score'], bins=bins, labels=labels, right=False)
        score_buckets = closed_df.groupby('score_range', observed=False).agg(
            trades=('trade_id', 'count'),
            wins=('is_accurate', lambda x: (x == True).sum()),
            losses=('is_accurate', lambda x: (x == False).sum()),
            avg_profit=('profit', 'mean'),
        ).reset_index()
        score_buckets['win_rate'] = score_buckets.apply(
            lambda r: round(r['wins'] / (r['wins'] + r['losses']) * 100, 1)
            if (r['wins'] + r['losses']) > 0 else 0, axis=1
        )

    return {
        'total_signals': total,
        'traded': int(traded),
        'closed': int(closed),
        'accurate': int(accurate),
        'inaccurate': int(inaccurate),
        'overall_accuracy': round(accurate / (accurate + inaccurate) * 100, 1)
            if (accurate + inaccurate) > 0 else 0,
        'by_signal': by_signal,
        'by_score': score_buckets,
    }


def save_verification_result(
    signal_id: int, symbol: str, name: str, signal: str, score: int,
    signal_price: float, verify_price: float, change_pct: float,
    is_correct: bool, is_flat: bool,
    entry_price: float = None, stop_loss: float = None,
    target_1: float = None, target_2: float = None,
):
    """Save a single signal verification result."""
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d")

    # Get contract multiplier (strip trailing numbers from symbol like RB0 -> RB)
    base = symbol.rstrip("0123456789")
    mult = CONTRACT_MULTIPLIERS.get(base, 10)

    # Calculate profit per lot
    if is_flat:
        profit = 0
    elif is_correct:
        profit = abs(verify_price - signal_price) * mult
    else:
        profit = -abs(verify_price - signal_price) * mult

    try:
        conn.execute(
            """INSERT OR REPLACE INTO verification_log
               (signal_id, verify_date, symbol, name, signal, score,
                signal_price, verify_price, change_pct, is_correct, is_flat,
                contract_mult, profit_per_lot, entry_price, stop_loss, target_1, target_2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal_id, now, symbol, name, signal, score,
             signal_price, verify_price, change_pct,
             1 if is_correct else 0, 1 if is_flat else 0,
             mult, round(profit, 2),
             entry_price, stop_loss, target_1, target_2),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Save verification failed for {symbol}: {e}")
    finally:
        conn.close()


def get_verification_daily(date: str = None) -> pd.DataFrame:
    """Get verification results for a specific date (default: today)."""
    conn = get_connection()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    try:
        df = pd.read_sql_query(
            """SELECT * FROM verification_log
               WHERE verify_date = ?
               ORDER BY signal_id DESC""",
            conn, params=(date,),
        )
        return df
    except Exception as e:
        logger.error(f"Verification query failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_verification_summary(date: str = None) -> dict:
    """Summary stats for daily verification."""
    df = get_verification_daily(date)
    if df.empty:
        return {}

    total = len(df)
    correct = int((df['is_correct'] == 1).sum())
    wrong = int(((df['is_correct'] == 0) & (df['is_flat'] == 0)).sum())
    flat = int((df['is_flat'] == 1).sum())
    meaningful = correct + wrong

    total_profit = float(df['profit_per_lot'].sum())
    avg_profit = float(df[df['is_flat'] == 0]['profit_per_lot'].mean()) if meaningful > 0 else 0

    # By signal type
    by_sig = df.groupby('signal').agg(
        total=('id', 'count'),
        correct=('is_correct', 'sum'),
        wrong=('is_correct', lambda x: ((x == 0) & (df.loc[x.index, 'is_flat'] == 0)).sum()),
        flat=('is_flat', 'sum'),
        profit=('profit_per_lot', 'sum'),
    ).reset_index()

    if not by_sig.empty:
        by_sig['accuracy'] = by_sig.apply(
            lambda r: round(r['correct'] / (r['correct'] + r['wrong']) * 100, 1)
            if (r['correct'] + r['wrong']) > 0 else 0, axis=1
        )

    # Accuracy by score range (excl flat)
    non_flat = df[df['is_flat'] == 0].copy()
    by_score = pd.DataFrame()
    if not non_flat.empty:
        bins = [0, 30, 40, 50, 60, 100]
        labels = ['0-30', '30-40', '40-50', '50-60', '60+']
        non_flat['score_range'] = pd.cut(non_flat['score'], bins=bins, labels=labels, right=False)
        by_score = non_flat.groupby('score_range', observed=False).agg(
            total=('id', 'count'),
            correct=('is_correct', 'sum'),
            profit=('profit_per_lot', 'sum'),
        ).reset_index()
        by_score['accuracy'] = by_score.apply(
            lambda r: round(r['correct'] / r['total'] * 100, 1) if r['total'] > 0 else 0, axis=1
        )

    return {
        'date': date or datetime.now().strftime("%Y-%m-%d"),
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'flat': flat,
        'meaningful': meaningful,
        'accuracy': round(correct / meaningful * 100, 1) if meaningful > 0 else 0,
        'total_profit': round(total_profit, 2),
        'avg_profit': round(avg_profit, 2),
        'by_signal': by_sig,
        'by_score': by_score,
    }
