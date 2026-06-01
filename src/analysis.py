"""
Analysis functions. Each function returns a figure and/or a DataFrame.
Charts are saved to output/charts/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

CHARTS = Path(__file__).parent.parent / "output" / "charts"
TABLES = Path(__file__).parent.parent / "output" / "tables"
CHARTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# Color palette
PALETTE = {"Fear": "#E63946", "Neutral": "#F4A261", "Greed": "#2A9D8F"}
PALETTE5 = {
    "Extreme Fear": "#9B2226",
    "Fear":         "#E63946",
    "Neutral":      "#F4A261",
    "Greed":        "#2A9D8F",
    "Extreme Greed":"#264653",
}

sns.set_theme(style="whitegrid", font_scale=1.1)

# CHART 1: PnL distribution by sentiment

def chart_pnl_by_sentiment(merged: pd.DataFrame) -> plt.Figure:
    closes = merged[merged["is_close"] & merged["closed_pnl"].abs().lt(5000)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trade PnL Distribution by Market Sentiment", fontsize=14, fontweight="bold")

    # Box plot
    order = ["Fear", "Neutral", "Greed"]
    colors = [PALETTE[s] for s in order]
    bp = axes[0].boxplot(
        [closes.loc[closes["sentiment_binary"] == s, "closed_pnl"].dropna() for s in order],
        labels=order, patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Closed PnL Distribution (no outliers)")
    axes[0].set_ylabel("Closed PnL (USD)")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Mean PnL + CI bars
    rows = []
    for s in order:
        vals = closes.loc[closes["sentiment_binary"] == s, "closed_pnl"].dropna()
        ci = stats.sem(vals) * stats.t.ppf(0.975, len(vals)-1)
        rows.append({"Sentiment": s, "Mean PnL": vals.mean(), "CI": ci})
    df_bar = pd.DataFrame(rows)
    bars = axes[1].bar(df_bar["Sentiment"], df_bar["Mean PnL"],
                       yerr=df_bar["CI"], color=colors, alpha=0.8,
                       capsize=5, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Mean Closed PnL per Trade ± 95% CI")
    axes[1].set_ylabel("Mean Closed PnL (USD)")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, mean_val in zip(bars, df_bar["Mean PnL"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"${mean_val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(CHARTS / "01_pnl_by_sentiment.png", dpi=150, bbox_inches="tight")
    return fig

# CHART 2: Win rate by sentiment

def chart_winrate_by_sentiment(merged: pd.DataFrame) -> plt.Figure:
    closes = merged[merged["is_close"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Win Rate & Trade Volume by Market Sentiment", fontsize=14, fontweight="bold")

    order = ["Fear", "Neutral", "Greed"]
    colors = [PALETTE[s] for s in order]

    # Win rate per day
    daily_wr = (
        closes.groupby(["date", "sentiment_binary"])["closed_pnl"]
        .apply(lambda x: (x > 0).mean())
        .reset_index(name="win_rate")
    )
    bp = axes[0].boxplot(
        [daily_wr.loc[daily_wr["sentiment_binary"] == s, "win_rate"].dropna() for s in order],
        labels=order, patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Daily Win Rate Distribution")
    axes[0].set_ylabel("Win Rate")
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    axes[0].legend(fontsize=8)

    # Trades per day
    daily_vol = (
        merged.groupby(["date", "sentiment_binary"])["account"]
        .count()
        .reset_index(name="n_trades")
    )
    bp2 = axes[1].boxplot(
        [daily_vol.loc[daily_vol["sentiment_binary"] == s, "n_trades"].dropna() for s in order],
        labels=order, patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Trades Per Day")
    axes[1].set_ylabel("Number of Trades")

    plt.tight_layout()
    fig.savefig(CHARTS / "02_winrate_volume_by_sentiment.png", dpi=150, bbox_inches="tight")
    return fig

# CHART 3: Behavior shifts (leverage proxy, size, long/short)

def chart_behavior_by_sentiment(merged: pd.DataFrame) -> plt.Figure:
    order = ["Fear", "Neutral", "Greed"]
    colors = [PALETTE[s] for s in order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Trader Behavior Changes by Market Sentiment", fontsize=14, fontweight="bold")

    # 1. Average trade size
    size_data = [merged.loc[merged["sentiment_binary"] == s, "size_usd"].dropna() for s in order]
    bp1 = axes[0].boxplot(size_data, labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_title("Trade Size (USD)")
    axes[0].set_ylabel("Size USD")

    # 2. Long ratio per sentiment
    long_ratio = (
        merged[merged["is_long"] | merged["is_short"]]
        .groupby("sentiment_binary")
        .apply(lambda x: x["is_long"].sum() / len(x))
        .reindex(order)
    )
    axes[1].bar(order, long_ratio.values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Long Trade Ratio")
    axes[1].set_ylabel("Fraction of Directional Trades → Long")
    for i, v in enumerate(long_ratio.values):
        axes[1].text(i, v + 0.005, f"{v:.2%}", ha="center", fontsize=9)

    # 3. Avg size per day across sentiment (violin)
    daily_size = (
        merged.groupby(["date", "sentiment_binary"])["size_usd"]
        .mean()
        .reset_index(name="avg_size")
    )
    vp_data = [daily_size.loc[daily_size["sentiment_binary"] == s, "avg_size"].dropna() for s in order]
    vp = axes[2].violinplot(vp_data, positions=range(len(order)), showmedians=True)
    for i, (body, color) in enumerate(zip(vp["bodies"], colors)):
        body.set_facecolor(color); body.set_alpha(0.6)
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order)
    axes[2].set_title("Daily Avg Trade Size Distribution")
    axes[2].set_ylabel("Avg USD Size per Day")

    plt.tight_layout()
    fig.savefig(CHARTS / "03_behavior_by_sentiment.png", dpi=150, bbox_inches="tight")
    return fig

# CHART 4: Trader segments heatmap

def chart_segment_heatmap(trader_df: pd.DataFrame, segments_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of PnL / win rate across trader segments × sentiment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trader Segments: PnL & Win Rate by Sentiment", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes,
                                  ["total_pnl", "win_rate"],
                                  ["Total PnL (USD)", "Win Rate"]):
        pivot = segments_df.pivot_table(
            index="segment", columns="sentiment_binary", values=metric, aggfunc="mean"
        )
        # Reorder columns
        for col in ["Greed", "Neutral", "Fear"]:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot = pivot[["Fear", "Neutral", "Greed"]]
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Trader Segment")

    plt.tight_layout()
    fig.savefig(CHARTS / "04_segment_heatmap.png", dpi=150, bbox_inches="tight")
    return fig

# CHART 5: Cumulative PnL time-series

def chart_cumulative_pnl(daily_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    daily_sorted = daily_df.sort_values("date")
    daily_sorted["cum_pnl"] = daily_sorted["total_pnl"].cumsum()

    ax.plot(pd.to_datetime(daily_sorted["date"]), daily_sorted["cum_pnl"],
            color="#264653", linewidth=1.5, label="Cumulative PnL")

    # Shade by sentiment
    sentiment_colors = {"Fear": "#E63946", "Neutral": "#F4A261", "Greed": "#2A9D8F"}
    for _, row in daily_sorted.iterrows():
        color = sentiment_colors.get(row["sentiment_binary"], "grey")
        ax.axvspan(pd.Timestamp(row["date"]) - pd.Timedelta(hours=12),
                   pd.Timestamp(row["date"]) + pd.Timedelta(hours=12),
                   alpha=0.08, color=color, linewidth=0)

    patches = [mpatches.Patch(color=c, label=s, alpha=0.4)
               for s, c in sentiment_colors.items()]
    ax.legend(handles=[ax.lines[0]] + patches,
              labels=["Cumulative PnL"] + list(sentiment_colors.keys()), fontsize=9)
    ax.set_title("Cumulative PnL Over Time (shaded by daily sentiment)", fontsize=13)
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(CHARTS / "05_cumulative_pnl.png", dpi=150, bbox_inches="tight")
    return fig

# CHART 6: Segment profiles

def chart_cluster_profiles(trader_df: pd.DataFrame) -> plt.Figure:
    """Radar / bar chart of cluster characteristics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Trader Segment Profiles", fontsize=14, fontweight="bold")

    seg_summary = trader_df.groupby("segment")[
        ["win_rate", "trades_per_day", "avg_size_usd", "net_pnl"]
    ].mean().reset_index()

    # Shorten labels so they fit
    label_map = {
        "Aggressive Momentum":      "Aggressive\nMomentum",
        "Cautious Scalpers":        "Cautious\nScalpers",
        "High-Volume Traders":      "High-Volume\nTraders",
        "Swing / Position Traders": "Swing /\nPosition Traders",
    }
    short_labels = seg_summary["segment"].map(label_map).fillna(seg_summary["segment"])

    metrics = ["win_rate", "trades_per_day", "avg_size_usd", "net_pnl"]
    titles  = ["Win Rate", "Trades / Day", "Avg Trade Size (USD)", "Net PnL (USD)"]
    colors  = ["#E63946", "#F4A261", "#2A9D8F", "#264653"]

    for ax, metric, title, color in zip(axes[:3], metrics[:3], titles[:3], colors[:3]):
        bars = ax.bar(short_labels, seg_summary[metric],
                      color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # Net PnL
    axes[2].bar(short_labels, seg_summary["net_pnl"],
                color=["#E63946" if v < 0 else "#2A9D8F" for v in seg_summary["net_pnl"]],
                alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("Net PnL by Segment (USD)", fontsize=11)
    axes[2].tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    fig.savefig(CHARTS / "06_cluster_profiles.png", dpi=150, bbox_inches="tight")
    return fig

# STATS TABLE

def sentiment_stats_table(merged: pd.DataFrame) -> pd.DataFrame:
    closes = merged[merged["is_close"]].copy()
    rows = []
    for s in ["Fear", "Neutral", "Greed"]:
        sub = closes[closes["sentiment_binary"] == s]["closed_pnl"].dropna()
        rows.append({
            "Sentiment":    s,
            "N_trades":     len(sub),
            "Mean_PnL":     round(sub.mean(), 4),
            "Median_PnL":   round(sub.median(), 4),
            "Win_Rate":     round((sub > 0).mean(), 4),
            "Std_PnL":      round(sub.std(), 4),
            "Total_PnL":    round(sub.sum(), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "sentiment_pnl_stats.csv", index=False)
    return df

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_prep import load_fear_greed, load_trades, merge_datasets
    from metrics import daily_metrics, trader_metrics

    fg = load_fear_greed()
    ht = load_trades()
    merged = merge_datasets(ht, fg)
    dm = daily_metrics(merged)
    tm = trader_metrics(merged)

    chart_pnl_by_sentiment(merged)
    chart_winrate_by_sentiment(merged)
    chart_behavior_by_sentiment(merged)
    chart_cumulative_pnl(dm)
    print("Charts saved to output/charts/")