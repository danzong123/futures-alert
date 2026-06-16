"""
Trade Tracker CLI - Record and manage futures trades.

Usage:
  python trade_tracker.py add --symbol RB0 --name "螺纹钢" --direction long --entry 3200
  python trade_tracker.py close --id 1 --exit 3300
  python trade_tracker.py list [--status open|closed]
  python trade_tracker.py summary
  python trade_tracker.py delete --id 1
"""
import os, sys, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import (
    init_db, add_trade, close_trade, get_all_trades, get_performance_summary, delete_trade,
    get_signal_history
)


def cmd_add(args):
    trade_id = add_trade(args.symbol, args.name, args.direction, args.entry, args.quantity, args.signal_id)
    if trade_id:
        print(f"Trade #{trade_id} opened: {args.symbol} {args.direction} @ {args.entry}")
    else:
        print("Failed to add trade")


def cmd_close(args):
    ok = close_trade(args.id, args.exit)
    if ok:
        print(f"Trade #{args.id} closed @ {args.exit}")
    else:
        print(f"Trade #{args.id} not found or already closed")


def cmd_list(args):
    df = get_all_trades(args.status)
    if df.empty:
        print("No trades found")
        return

    cols = ['id', 'symbol', 'name', 'direction', 'entry_price', 'exit_price', 'profit', 'profit_pct', 'status', 'entry_date', 'exit_date']
    show = [c for c in cols if c in df.columns]
    print(df[show].to_string(index=False))


def cmd_summary(args):
    s = get_performance_summary()
    if not s or s.get('total_trades', 0) == 0:
        print("No closed trades yet")
        return

    print("=" * 55)
    print("  TRADE PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"  Total Trades:   {s['total_trades']}")
    print(f"  Open Positions: {s['open_trades']}")
    print(f"  Wins / Losses:  {s['wins']} / {s['losses']}")
    print(f"  Win Rate:       {s['win_rate']:.1f}%")
    print(f"  Total PnL:      {s['total_pnl']:+.2f}")
    print(f"  Avg Profit:     {s['avg_profit_pct']:+.2f}%")
    if s.get('best_trade'):
        b = s['best_trade']
        print(f"  Best Trade:     {b['symbol']} {b['profit_pct']:+.2f}% ({b['profit']:+.2f})")
    if s.get('worst_trade'):
        w = s['worst_trade']
        print(f"  Worst Trade:    {w['symbol']} {w['profit_pct']:+.2f}% ({w['profit']:+.2f})")
    print()

    by_sym = s.get('by_symbol')
    if by_sym is not None and not by_sym.empty:
        print("-" * 55)
        print("  Per-Symbol Breakdown")
        print("-" * 55)
        cols = ['symbol', 'trades', 'wins', 'losses', 'win_rate', 'total_pnl', 'avg_pct']
        by_sym['win_rate'] = (by_sym['wins'] / by_sym['trades'] * 100).round(1)
        by_sym['win_rate'] = by_sym['win_rate'].apply(lambda x: f"{x:.0f}%")
        by_sym['total_pnl'] = by_sym['total_pnl'].round(2).apply(lambda x: f"{x:+.2f}")
        by_sym['avg_pct'] = by_sym['avg_pct'].round(2).apply(lambda x: f"{x:+.2f}%")
        by_sym = by_sym.rename(columns={
            'symbol': 'Symbol', 'trades': 'Trades', 'wins': 'W', 'losses': 'L',
            'win_rate': 'Win%', 'total_pnl': 'PnL', 'avg_pct': 'Avg%'
        })
        print(by_sym.to_string(index=False))


def cmd_delete(args):
    ok = delete_trade(args.id)
    print(f"Trade #{args.id} {'deleted' if ok else 'not found'}")


def main():
    parser = argparse.ArgumentParser(description="Trade Tracker")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="Record a new trade")
    p_add.add_argument("--symbol", required=True)
    p_add.add_argument("--name", default="")
    p_add.add_argument("--direction", choices=["long", "short"], default="long")
    p_add.add_argument("--entry", type=float, required=True)
    p_add.add_argument("--quantity", type=int, default=1)
    p_add.add_argument("--signal-id", type=int)

    p_close = sub.add_parser("close", help="Close a trade")
    p_close.add_argument("--id", type=int, required=True)
    p_close.add_argument("--exit", type=float, required=True)

    p_list = sub.add_parser("list", help="List trades")
    p_list.add_argument("--status", choices=["open", "closed"])

    p_summary = sub.add_parser("summary", help="Show performance summary")

    p_del = sub.add_parser("delete", help="Delete a trade")
    p_del.add_argument("--id", type=int, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    init_db()

    handlers = {
        "add": cmd_add,
        "close": cmd_close,
        "list": cmd_list,
        "summary": cmd_summary,
        "delete": cmd_delete,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
