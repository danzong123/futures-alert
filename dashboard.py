"""
Futures Alert Dashboard - Streamlit web interface.

Provides real-time views of signals, contract data, trades, and performance.
"""
import os
import sys
import time
from datetime import datetime

# Strip broken proxy settings before any network imports
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_key, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.database import (
    init_db, get_signal_history, get_latest_snapshots,
    get_all_trades, get_performance_summary, get_alert_history,
    get_signal_accuracy_summary,
)

st.set_page_config(
    page_title="Futures Alert",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid rgba(120, 120, 255, 0.15);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
    .bull { color: #00e676; }
    .bear { color: #ff1744; }
    .neutral { color: #ffab40; }
    [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; }
    div[data-testid="stMetric"] { background: transparent; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a14 0%, #111122 100%);
        border-right: 1px solid rgba(120,120,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ---- Init ----
init_db()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## \U0001f4c8 Futures Alert")
    st.caption("Real-time monitoring dashboard")

    st.divider()

    refresh_sec = st.slider("Auto-refresh (seconds)", 0, 120, 0, step=5,
                            help="Set to 0 to disable auto-refresh")
    if refresh_sec > 0:
        time.sleep(refresh_sec)
        st.rerun()

    st.divider()
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

# ---- Load Data ----
snapshots = get_latest_snapshots()
signals = get_signal_history(hours=24)
trades_df = get_all_trades()
perf = get_performance_summary()
alerts = get_alert_history(hours=24)

# ---- Header Row ----
col1, col2, col3, col4, col5 = st.columns(5)

bull_count = len(signals[signals["signal"] == "bull"]) if not signals.empty else 0
bear_count = len(signals[signals["signal"] == "bear"]) if not signals.empty else 0
contract_count = len(snapshots) if not snapshots.empty else 0
open_trades = perf.get("open_trades", 0)
total_pnl = perf.get("total_pnl", 0)

with col1:
    st.metric("Contracts", contract_count)
with col2:
    st.metric("\U0001f7e2 Bull Signals", bull_count)
with col3:
    st.metric("\U0001f534 Bear Signals", bear_count)
with col4:
    st.metric("Open Trades", open_trades)
with col5:
    val = f"{total_pnl:+,.2f}" if total_pnl else "\u2014"
    st.metric("Total P&L", val)

st.divider()

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f4ca Contract Data",
    "\U0001f514 Signals",
    "\U0001f4b0 Trades",
    "\U0001f4cb Alerts",
])

# ---- Tab 1: Contract Data ----
with tab1:
    if snapshots.empty:
        st.info("No contract data yet. Run `python main.py --now` to populate.")
    else:
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search = st.text_input("Search symbol or name", "",
                                   placeholder="e.g. RB, \u94dc, \u539f\u6cb9",
                                   key="contract_search")
        with col_f2:
            sort_col = st.selectbox("Sort by",
                                    ["symbol", "latest_price", "range_high", "range_low",
                                     "indicator_1_value", "indicator_2_value"],
                                    key="contract_sort")

        df = snapshots.copy()
        if search:
            mask = df["symbol"].str.lower().str.contains(search.lower()) | \
                   df["name"].str.lower().str.contains(search.lower())
            df = df[mask]

        df = df.sort_values(sort_col, ascending=(sort_col == "symbol"))

        # Format
        disp = df[["symbol", "name", "latest_price", "range_high", "range_low",
                    "indicator_1_name", "indicator_1_value",
                    "indicator_2_name", "indicator_2_value"]].copy()
        for c in ["latest_price", "range_high", "range_low", "indicator_1_value", "indicator_2_value"]:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "\u2014")

        disp.columns = ["Symbol", "Name", "Price", "Range High", "Range Low",
                        "Ind.1", "Ind.1 Val", "Ind.2", "Ind.2 Val"]

        st.dataframe(disp, use_container_width=True, height=600, hide_index=True)

        # Range chart
        if not df.empty:
            st.subheader("Price vs Range")
            chart_df = df.dropna(subset=["latest_price", "range_high", "range_low"]).head(30).copy()
            chart_df["pos"] = chart_df["latest_price"] - chart_df["range_low"]
            chart_df["pos_pct"] = ((chart_df["latest_price"] - chart_df["range_low"]) /
                                   (chart_df["range_high"] - chart_df["range_low"]) * 100)

            fig = px.bar(
                chart_df.sort_values("pos_pct"),
                x="symbol", y="pos_pct",
                color="pos_pct",
                color_continuous_scale=["#ff1744", "#ffab40", "#00e676"],
                range_color=[0, 100],
                title="Position within Range (0% = low, 100% = high)",
            )
            fig.update_layout(
                xaxis_title=None, yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Signals ----
with tab2:
    if signals.empty:
        st.info("No signals in the last 24 hours.")
    else:
        # Summary chart
        if not signals.empty:
            sig_summary = signals.groupby("signal").size().reset_index(name="count")
            colors_map = {"bull": "#00e676", "bear": "#ff1744"}
            colors_list = [colors_map.get(s, "#888") for s in sig_summary["signal"]]

            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                fig = go.Figure(data=[go.Pie(
                    labels=sig_summary["signal"].str.upper(),
                    values=sig_summary["count"],
                    marker_colors=colors_list,
                    hole=0.5,
                )])
                fig.update_layout(
                    title="Signal Distribution (24h)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=300, margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_s2:
                st.subheader("Recent Signals")
                disp_sig = signals[["timestamp", "symbol", "name", "signal", "score", "price"]].head(20).copy()
                disp_sig["signal"] = disp_sig["signal"].apply(
                    lambda s: f"\U0001f7e2 {s.upper()}" if s == "bull" else f"\U0001f534 {s.upper()}"
                )
                disp_sig.columns = ["Time", "Symbol", "Name", "Signal", "Score", "Price"]
                st.dataframe(disp_sig, use_container_width=True, height=400, hide_index=True)

# ---- Tab 3: Trades ----
with tab3:
    if trades_df.empty:
        st.info("No trades recorded yet. Use `python trade_tracker.py add` to record.")
    else:
        st.subheader("Performance Summary")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Total Trades", perf.get("total_trades", 0))
        with tc2:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
        with tc3:
            st.metric("W / L", f"{perf.get('wins', 0)} / {perf.get('losses', 0)}")
        with tc4:
            st.metric("Avg P&L %", f"{perf.get('avg_profit_pct', 0):+.2f}%")

        st.divider()

        # Trades table
        open_trades_df = trades_df[trades_df["status"] == "open"]
        closed_trades_df = trades_df[trades_df["status"] == "closed"]

        if not open_trades_df.empty:
            st.subheader(f"Open Positions ({len(open_trades_df)})")
            disp_open = open_trades_df[["symbol", "name", "direction", "entry_price", "entry_date", "quantity"]].copy()
            disp_open["direction"] = disp_open["direction"].str.upper()
            disp_open.columns = ["Symbol", "Name", "Dir", "Entry", "Date", "Qty"]
            st.dataframe(disp_open, use_container_width=True, hide_index=True)

        if not closed_trades_df.empty:
            st.subheader(f"Closed Trades ({len(closed_trades_df)})")
            disp_closed = closed_trades_df[
                ["symbol", "name", "direction", "entry_price", "exit_price",
                 "profit", "profit_pct", "entry_date", "exit_date"]
            ].copy()
            disp_closed["direction"] = disp_closed["direction"].str.upper()
            disp_closed["profit"] = disp_closed["profit"].apply(
                lambda x: f"{x:+,.2f}" if pd.notna(x) else "\u2014"
            )
            disp_closed["profit_pct"] = disp_closed["profit_pct"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "\u2014"
            )
            disp_closed.columns = ["Symbol", "Name", "Dir", "Entry", "Exit",
                                   "P&L", "P&L%", "Entry Date", "Exit Date"]
            st.dataframe(disp_closed, use_container_width=True, hide_index=True)

            # P&L chart
            if len(closed_trades_df) >= 2:
                chart_trades = closed_trades_df.sort_values("entry_date").copy()
                chart_trades["cum_pnl"] = chart_trades["profit"].cumsum()
                fig = px.line(
                    chart_trades, x="entry_date", y="cum_pnl",
                    title="Cumulative P&L",
                    markers=True,
                )
                fig.update_traces(line_color="#00e676")
                fig.update_layout(
                    xaxis_title=None, yaxis_title="Cumulative P&L",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Alerts ----
with tab4:
    if alerts.empty:
        st.info("No alerts in the last 24 hours.")
    else:
        st.subheader(f"Alert History ({len(alerts)} records)")
        disp_alerts = alerts[["timestamp", "title", "content", "channel"]].head(50).copy()
        disp_alerts.columns = ["Time", "Title", "Content", "Channel"]
        st.dataframe(disp_alerts, use_container_width=True, height=500, hide_index=True)

    # ---- Signal Accuracy Analysis ----
    st.divider()
    st.subheader("📊 Signal Accuracy Analysis")

    analysis = get_signal_accuracy_summary(hours=168)

    if not analysis:
        st.info("Not enough data yet. Signals and trades will accumulate over time. Record trades with `python trade_tracker.py add --signal-id <ID>` to link them to signals.")
    else:
        # ---- Row 1: Summary Metrics ----
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.metric("Total Signals (7d)", analysis.get("total_signals", 0))
        with mc2:
            traded = analysis.get("traded", 0)
            total = max(analysis.get("total_signals", 1), 1)
            st.metric("Signals Traded", f"{traded} ({traded/total*100:.0f}%)")
        with mc3:
            st.metric("Trades Closed", analysis.get("closed", 0))
        with mc4:
            st.metric("Accurate", analysis.get("accurate", 0))
        with mc5:
            acc = analysis.get("overall_accuracy", 0)
            st.metric("Accuracy", f"{acc:.0f}%" if acc else "—")

        # ---- Row 2: By Signal Type ----
        st.divider()
        by_sig = analysis.get("by_signal")
        if by_sig is not None and not by_sig.empty:
            col_sa1, col_sa2 = st.columns(2)

            with col_sa1:
                st.markdown("**Accuracy by Signal Type**")
                fig = go.Figure()
                colors_sig = {"bull": "#00e676", "bear": "#ff1744"}
                for _, row in by_sig.iterrows():
                    sig = row["signal"]
                    fig.add_trace(go.Bar(
                        name=sig.upper(),
                        x=[sig.upper()],
                        y=[row["accuracy_pct"]],
                        marker_color=colors_sig.get(sig, "#888"),
                        text=f'{row["accuracy_pct"]}%',
                        textposition="outside",
                    ))
                fig.update_layout(
                    yaxis=dict(title="Accuracy %", range=[0, 100]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=350, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_sa2:
                st.markdown("**Profit by Signal Type**")
                fig2 = go.Figure()
                for _, row in by_sig.iterrows():
                    sig = row["signal"]
                    fig2.add_trace(go.Bar(
                        name=sig.upper(),
                        x=[sig.upper()],
                        y=[row["total_profit"] if pd.notna(row["total_profit"]) else 0],
                        marker_color=colors_sig.get(sig, "#888"),
                        text=f'{row["total_profit"]:+,.0f}' if pd.notna(row["total_profit"]) else "0",
                        textposition="outside",
                    ))
                fig2.update_layout(
                    yaxis=dict(title="Total P&L"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=350, showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ---- Row 3: Score vs Outcome ----
        by_score = analysis.get("by_score")
        if by_score is not None and not by_score.empty:
            st.divider()
            st.markdown("**Win Rate by Score Range** (Does a higher score correlate with better results?)")
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=by_score["score_range"],
                y=by_score["win_rate"],
                marker=dict(
                    color=by_score["win_rate"].apply(
                        lambda v: "#00e676" if v >= 60 else ("#ffab40" if v >= 40 else "#ff1744")
                    ),
                ),
                text=by_score["win_rate"].apply(lambda v: f"{v:.0f}%"),
                textposition="outside",
            ))
            fig3.update_layout(
                xaxis=dict(title="Score Range"),
                yaxis=dict(title="Win Rate %", range=[0, 105]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)

            disp_score = by_score.copy()
            disp_score["avg_profit"] = disp_score["avg_profit"].apply(
                lambda x: f"{x:+,.2f}" if pd.notna(x) else "—"
            )
            disp_score.columns = ["Score Range", "Trades", "Wins", "Losses", "Avg P&L", "Win Rate"]
            st.dataframe(disp_score, use_container_width=True, hide_index=True)

        # ---- Important note ----
        st.caption(
            "**How to link trades to signals:** When recording a trade triggered by a signal, "
            "use `python trade_tracker.py add --symbol XX --name X --direction long/short "
            "--entry XXXX --signal-id <ID>`. This connects the trade to the signal for accuracy tracking."
        )

# ---- Footer ----
# ---- Footer ----
st.divider()
st.caption("\U0001f3e6 Futures Alert System \u00b7 Dashboard \u00b7 All data from local SQLite")
