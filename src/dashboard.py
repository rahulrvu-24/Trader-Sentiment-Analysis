"""
Lightweight Streamlit dashboard for Primetrade Assignment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import load_fear_greed, load_trades, merge_datasets
from metrics import daily_metrics, trader_metrics, sentiment_behavior_summary
from models import cluster_traders

st.set_page_config(
    page_title="Sentiment vs Trader Performance",
    page_icon="📈",
    layout="wide",
)

PALETTE = {"Fear": "#E63946", "Neutral": "#F4A261", "Greed": "#2A9D8F"}


#  LOAD DATA (cached)

@st.cache_data
def load_all():
    processed = Path(__file__).parent.parent / "data" / "processed"
    merged_path = processed / "trades_with_sentiment.csv"
    if merged_path.exists():
        merged = pd.read_csv(merged_path, parse_dates=["datetime"])
        merged["date"] = pd.to_datetime(merged["date"]).dt.date
        merged["is_close"]  = merged["direction"].str.startswith("Close")
        merged["net_pnl"]   = merged["closed_pnl"] - merged["fee"]
        merged["is_long"]   = merged["direction"].str.contains("Long", na=False)
        merged["is_short"]  = merged["direction"].str.contains("Short", na=False)
    else:
        fg = load_fear_greed()
        ht = load_trades()
        merged = merge_datasets(ht, fg)
    dm = daily_metrics(merged)
    tm = trader_metrics(merged)
    tm_cl = cluster_traders(tm, n_clusters=4)
    return merged, dm, tm_cl

merged, dm, tm_cl = load_all()

#  SIDEBAR FILTERS

st.sidebar.title("🔍 Filters")
sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=["Fear", "Neutral", "Greed"],
    default=["Fear", "Neutral", "Greed"],
)

all_accounts = sorted(merged["account"].unique().tolist())
short_accounts = {a: a[:10] + "…" for a in all_accounts}
selected_accs = st.sidebar.multiselect(
    "Accounts (optional)",
    options=all_accounts,
    format_func=lambda x: short_accounts[x],
    default=[],
)

date_min = pd.to_datetime(merged["date"].min())
date_max = pd.to_datetime(merged["date"].max())
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max,
)

# Apply filters
filt = merged["sentiment_binary"].isin(sentiments)
if selected_accs:
    filt &= merged["account"].isin(selected_accs)
if len(date_range) == 2:
    filt &= (pd.to_datetime(merged["date"]) >= pd.Timestamp(date_range[0]))
    filt &= (pd.to_datetime(merged["date"]) <= pd.Timestamp(date_range[1]))
df = merged[filt].copy()
closes = df[df["is_close"]].copy()

#  HEADER

st.title("📊 Trader Performance vs Market Sentiment")
st.caption("Hyperliquid Traders × Bitcoin Fear/Greed Index")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", f"{len(df):,}")
col2.metric("Unique Accounts", df["account"].nunique())
col3.metric("Total Closed PnL", f"${closes['closed_pnl'].sum():,.0f}")
col4.metric("Overall Win Rate", f"{(closes['closed_pnl'] > 0).mean():.1%}")

st.divider()

#  TAB LAYOUT

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 PnL by Sentiment",
    "🎭 Behavior Shifts",
    "👥 Trader Segments",
    "🔮 Predictive Signals",
])

# ── TAB 1 ──
with tab1:
    st.subheader("Performance Metrics by Sentiment")

    col_a, col_b = st.columns(2)

    with col_a:
        # Mean PnL bar
        fig, ax = plt.subplots(figsize=(6, 4))
        order = [s for s in ["Fear", "Neutral", "Greed"] if s in sentiments]
        means, errs, colors = [], [], []
        for s in order:
            vals = closes.loc[closes["sentiment_binary"] == s, "closed_pnl"]
            means.append(vals.mean())
            errs.append(vals.sem() * 1.96)
            colors.append(PALETTE[s])
        bars = ax.bar(order, means, yerr=errs, color=colors, alpha=0.8,
                      capsize=6, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Mean Closed PnL per Trade (±95% CI)")
        ax.set_ylabel("USD")
        st.pyplot(fig)
        plt.close()

    with col_b:
        # Win rate
        fig, ax = plt.subplots(figsize=(6, 4))
        wr_vals = []
        for s in order:
            v = closes.loc[closes["sentiment_binary"] == s, "closed_pnl"]
            wr_vals.append((v > 0).mean())
        colors_wr = [PALETTE[s] for s in order]
        ax.bar(order, wr_vals, color=colors_wr, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_title("Win Rate by Sentiment")
        ax.set_ylabel("Win Rate")
        for i, v in enumerate(wr_vals):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
        st.pyplot(fig)
        plt.close()

    # Stats table
    rows = []
    for s in order:
        v = closes.loc[closes["sentiment_binary"] == s, "closed_pnl"]
        rows.append({
            "Sentiment": s, "N Trades": len(v),
            "Mean PnL": f"${v.mean():.2f}", "Median PnL": f"${v.median():.2f}",
            "Win Rate": f"{(v>0).mean():.1%}", "Total PnL": f"${v.sum():,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Time series
    st.subheader("Cumulative PnL Over Time")
    dm_filt = dm[dm["sentiment_binary"].isin(sentiments)].sort_values("date").copy()
    dm_filt["cum_pnl"] = dm_filt["total_pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(dm_filt["date"]), dm_filt["cum_pnl"], color="#264653", linewidth=1.5)
    for _, row in dm_filt.iterrows():
        color = PALETTE.get(row["sentiment_binary"], "grey")
        ax.axvspan(pd.Timestamp(row["date"]) - pd.Timedelta(hours=12),
                   pd.Timestamp(row["date"]) + pd.Timedelta(hours=12),
                   alpha=0.07, color=color, linewidth=0)
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.set_xlabel("Date")
    ax.set_title("Cumulative PnL (shaded by daily sentiment)")
    st.pyplot(fig)
    plt.close()

# ── TAB 2 ──
with tab2:
    st.subheader("Behavioral Changes Across Sentiment Regimes")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        order_f = [s for s in ["Fear", "Neutral", "Greed"] if s in sentiments]
        long_ratios = [
            df[df["sentiment_binary"] == s]["is_long"].mean()
            for s in order_f
        ]
        ax.bar(order_f, long_ratios, color=[PALETTE[s] for s in order_f],
               alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Long Trade Ratio")
        ax.set_ylim(0, 1)
        for i, v in enumerate(long_ratios):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        avg_sizes = [df[df["sentiment_binary"] == s]["size_usd"].mean() for s in order_f]
        ax.bar(order_f, avg_sizes, color=[PALETTE[s] for s in order_f],
               alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title("Avg Trade Size (USD)")
        for i, v in enumerate(avg_sizes):
            ax.text(i, v + 50, f"${v:,.0f}", ha="center", fontsize=8)
        st.pyplot(fig); plt.close()

    with col_c:
        daily_vol = df.groupby(["date", "sentiment_binary"])["account"].count().reset_index(name="n_trades")
        fig, ax = plt.subplots(figsize=(5, 4))
        data_bp = [daily_vol[daily_vol["sentiment_binary"] == s]["n_trades"] for s in order_f]
        bp = ax.boxplot(data_bp, labels=order_f, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], order_f):
            patch.set_facecolor(PALETTE[s]); patch.set_alpha(0.7)
        ax.set_title("Trades Per Day")
        st.pyplot(fig); plt.close()

    # Detailed table
    st.subheader("Summary Table")
    sb_table = []
    for s in order_f:
        sub  = df[df["sentiment_binary"] == s]
        cl   = closes[closes["sentiment_binary"] == s]
        days = sub["date"].nunique()
        sb_table.append({
            "Sentiment": s, "Days": days,
            "Total Trades": len(sub),
            "Trades/Day": f"{len(sub)/max(days,1):.1f}",
            "Avg Size USD": f"${sub['size_usd'].mean():,.0f}",
            "Long %": f"{sub['is_long'].mean():.1%}",
            "Short %": f"{sub['is_short'].mean():.1%}",
            "Win Rate": f"{(cl['closed_pnl']>0).mean():.1%}",
        })
    st.dataframe(pd.DataFrame(sb_table), use_container_width=True, hide_index=True)

# ── TAB 3 ──
with tab3:
    st.subheader("Trader Behavioral Archetypes (K-Means, k=4)")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        seg_summary = tm_cl.groupby("segment")[
            ["win_rate", "net_pnl", "trades_per_day", "avg_size_usd", "long_bias"]
        ].mean().round(3).reset_index()
        seg_summary["net_pnl"] = seg_summary["net_pnl"].map(lambda x: f"${x:,.0f}")
        seg_summary["win_rate"] = seg_summary["win_rate"].map(lambda x: f"{x:.1%}")
        seg_summary["long_bias"] = seg_summary["long_bias"].map(lambda x: f"{x:.1%}")
        st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    with col_b:
        # PCA scatter
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_seg = {"Aggressive Momentum": "#E63946", "Cautious Scalpers": "#264653",
                      "High-Volume Traders": "#F4A261", "Swing / Position Traders": "#2A9D8F"}
        for seg, grp in tm_cl.groupby("segment"):
            ax.scatter(grp["pca1"], grp["pca2"], label=seg,
                       color=colors_seg.get(seg, "grey"), s=80, alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.set_title("PCA Projection of Trader Clusters")
        ax.legend(fontsize=7, loc="best")
        st.pyplot(fig); plt.close()

    # Segment × Sentiment table
    processed = Path(__file__).parent.parent / "output" / "tables" / "segment_by_sentiment.csv"
    if processed.exists():
        seg_sent = pd.read_csv(processed)
        st.subheader("Segment Performance by Sentiment")
        pivot = seg_sent.pivot_table(index="segment", columns="sentiment_binary",
                                     values=["total_pnl", "win_rate"], aggfunc="mean")
        st.dataframe(pivot.round(3), use_container_width=True)

# ── TAB 4 ──
with tab4:
    st.subheader("Key Predictive Signals")

    fi_path = Path(__file__).parent.parent / "output" / "tables" / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("**Feature Importances (Random Forest)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(fi["feature"][::-1], fi["rf_importance"][::-1],
                    color="#2A9D8F", alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.set_title("RF Feature Importance")
            st.pyplot(fig); plt.close()
        with col_b:
            st.dataframe(fi[["feature", "rf_importance", "gb_importance"]].round(4),
                         use_container_width=True, hide_index=True)

        st.info("""
        **Key findings from the model:**
        - Rolling 3-day PnL and prior win rate are the strongest predictors of next-day profitability.
        - **Sentiment features rank last** — the market regime alone doesn't predict next-day outcomes;
          it's *how traders behaved recently* that matters more.
        - This suggests sentiment is a *context* signal, not a direct *causal* driver of performance.
        """)

    # Strategy recommendations
    st.subheader("💡 Strategy Recommendations")
    st.markdown("""
    **Rule 1 — Size down on Fear days (for Aggressive Momentum traders)**
    > Fear days see 57% higher trade size vs Greed days for Aggressive Momentum traders,
    > yet their win rate drops ~17pp. Capping position size to ≤ 75% of usual during Fear
    > is expected to improve risk-adjusted returns for this segment.

    **Rule 2 — Lean long on Fear days, short on Greed**
    > Long trade ratio is 58% on Fear days vs 33% on Greed days — the market data confirms
    > that the crowd buys the dip on Fear. High-Volume Traders maintain 90%+ win rates on
    > Greed days with short bias; emulate this with a sentiment-conditional directional tilt.
    """)