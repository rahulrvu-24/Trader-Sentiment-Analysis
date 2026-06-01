"""
Feature engineering: daily aggregates, per-trader metrics, behavioral features.
"""

import pandas as pd
import numpy as np


#  DAILY METRICS  (per day × sentiment)

def daily_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade-level data to daily level.
    Only closing trades carry realized PnL; open trades are counted for volume.
    """
    closes = merged[merged["is_close"]].copy()

    # Daily PnL / win stats from closing trades
    daily_pnl = (
        closes.groupby(["date", "classification", "sentiment_binary"])
        .agg(
            total_pnl    = ("closed_pnl", "sum"),
            net_pnl      = ("net_pnl",    "sum"),
            trades_closed= ("closed_pnl", "count"),
            wins         = ("closed_pnl", lambda x: (x > 0).sum()),
            losses       = ("closed_pnl", lambda x: (x < 0).sum()),
            avg_win      = ("closed_pnl", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            avg_loss     = ("closed_pnl", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
            avg_size_usd = ("size_usd",   "mean"),
        )
        .reset_index()
    )
    daily_pnl["win_rate"] = daily_pnl["wins"] / daily_pnl["trades_closed"]

    # All trades for frequency / volume
    daily_vol = (
        merged.groupby("date")
        .agg(
            total_trades = ("account", "count"),
            unique_accounts = ("account", "nunique"),
            long_trades  = ("is_long",  "sum"),
            short_trades = ("is_short", "sum"),
            total_volume_usd = ("size_usd", "sum"),
        )
        .reset_index()
    )

    daily = daily_pnl.merge(daily_vol, on="date", how="left")
    daily["long_short_ratio"] = daily["long_trades"] / (daily["short_trades"] + 1e-9)

    # Drawdown proxy: cumulative daily net_pnl rolling max drawdown
    daily = daily.sort_values("date").reset_index(drop=True)
    roll_max = daily["net_pnl"].cummax()
    daily["drawdown_proxy"] = daily["net_pnl"] - roll_max

    return daily

#  PER-TRADER METRICS  (overall lifetime)

def trader_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-account aggregate: win rate, PnL, volume, leverage proxy, frequency."""
    closes = merged[merged["is_close"]].copy()

    trader_pnl = (
        closes.groupby("account")
        .agg(
            total_pnl     = ("closed_pnl", "sum"),
            net_pnl       = ("net_pnl",    "sum"),
            trades_closed = ("closed_pnl", "count"),
            wins          = ("closed_pnl", lambda x: (x > 0).sum()),
            losses        = ("closed_pnl", lambda x: (x < 0).sum()),
            avg_trade_pnl = ("closed_pnl", "mean"),
            std_pnl       = ("closed_pnl", "std"),
            avg_size_usd  = ("size_usd",   "mean"),
            total_fees    = ("fee",        "sum"),
        )
        .reset_index()
    )
    trader_pnl["win_rate"] = trader_pnl["wins"] / trader_pnl["trades_closed"]

    # Leverage proxy: avg position size / avg USD size (larger ratio -> higher notional per dollar)
    # We use avg size_tokens / avg size_usd as a rough "size per price unit" signal,
    # but the cleaner lever proxy is std_pnl / avg_size_usd (volatility-to-size)
    trader_pnl["pnl_volatility"] = trader_pnl["std_pnl"].fillna(0)
    trader_pnl["leverage_proxy"] = trader_pnl["avg_size_usd"] / (trader_pnl["avg_trade_pnl"].abs() + 1e-9)

    # Trade frequency: trades per active day
    active_days = (
        merged.groupby("account")["date"]
        .nunique()
        .rename("active_days")
        .reset_index()
    )
    all_trades = (
        merged.groupby("account")
        .agg(total_trades=("account", "count"))
        .reset_index()
    )

    trader = trader_pnl.merge(active_days, on="account").merge(all_trades, on="account")
    trader["trades_per_day"] = trader["total_trades"] / trader["active_days"]

    # Long/short bias
    long_count  = merged[merged["is_long"]].groupby("account").size().rename("n_long")
    short_count = merged[merged["is_short"]].groupby("account").size().rename("n_short")
    trader = trader.merge(long_count, on="account", how="left")
    trader = trader.merge(short_count, on="account", how="left")
    trader["n_long"]  = trader["n_long"].fillna(0)
    trader["n_short"] = trader["n_short"].fillna(0)
    trader["long_bias"] = trader["n_long"] / (trader["n_long"] + trader["n_short"] + 1e-9)

    return trader


#  PER-TRADER × SENTIMENT SPLIT

def trader_by_sentiment(merged: pd.DataFrame) -> pd.DataFrame:
    """Breakdown of trader performance by Fear vs Greed days."""
    closes = merged[merged["is_close"]].copy()

    result = (
        closes.groupby(["account", "sentiment_binary"])
        .agg(
            total_pnl     = ("closed_pnl", "sum"),
            trades        = ("closed_pnl", "count"),
            win_rate      = ("closed_pnl", lambda x: (x > 0).mean()),
            avg_size_usd  = ("size_usd",   "mean"),
        )
        .reset_index()
    )
    return result

#  DAILY BEHAVIOR × SENTIMENT

def sentiment_behavior_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Comparing trade behavior metrics across Fear / Neutral / Greed days.
    """
    closes = merged[merged["is_close"]].copy()

    summary = (
        merged.groupby("sentiment_binary")
        .agg(
            n_trades         = ("account", "count"),
            n_days           = ("date",    "nunique"),
            avg_size_usd     = ("size_usd", "mean"),
            long_pct         = ("is_long",  "mean"),
            short_pct        = ("is_short", "mean"),
        )
        .reset_index()
    )
    summary["trades_per_day"] = summary["n_trades"] / summary["n_days"]

    pnl_summary = (
        closes.groupby("sentiment_binary")
        .agg(
            avg_pnl          = ("closed_pnl", "mean"),
            win_rate         = ("closed_pnl", lambda x: (x > 0).mean()),
            avg_loss         = ("closed_pnl", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
        )
        .reset_index()
    )
    return summary.merge(pnl_summary, on="sentiment_binary", how="left")


if __name__ == "__main__":
    from data_prep import load_fear_greed, load_trades, merge_datasets
    fg = load_fear_greed()
    ht = load_trades()
    merged = merge_datasets(ht, fg)

    dm = daily_metrics(merged)
    print("Daily metrics shape:", dm.shape)
    print(dm.head(3).to_string())

    tm = trader_metrics(merged)
    print("\nTrader metrics shape:", tm.shape)
    print(tm.head(3).to_string())