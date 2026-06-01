"""
Data loading, cleaning, and merging for Primetrade Assignment.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# 1. LOAD

def load_fear_greed(path=None):
    """Load and clean the Bitcoin Fear/Greed Index."""
    path = path or RAW / "fear_greed_index.csv"
    fg = pd.read_csv(path)

    # Parse date
    fg["date"] = pd.to_datetime(fg["date"]).dt.date

    # Collapse Extreme Fear/Greed into binary label used in analysis
    fg["sentiment_binary"] = fg["classification"].map({
        "Extreme Fear": "Fear",
        "Fear":         "Fear",
        "Neutral":      "Neutral",
        "Greed":        "Greed",
        "Extreme Greed":"Greed",
    })

    fg = fg.sort_values("date").reset_index(drop=True)
    return fg


def load_trades(path=None):
    """Load and clean the Hyperliquid historical trader data."""
    path = path or RAW / "historical_data.csv"
    ht = pd.read_csv(path)

    # Parse timestamp (IST, format: DD-MM-YYYY HH:MM)
    ht["datetime"] = pd.to_datetime(ht["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    ht["date"]     = ht["datetime"].dt.date

    # Drop rows where timestamp couldn't be parsed
    before = len(ht)
    ht = ht.dropna(subset=["datetime"]).reset_index(drop=True)
    after = len(ht)
    if before != after:
        print(f"[data_prep] Dropped {before - after} rows with unparseable timestamps.")

    # Rename for convenience
    ht = ht.rename(columns={
        "Account":         "account",
        "Coin":            "coin",
        "Execution Price": "exec_price",
        "Size Tokens":     "size_tokens",
        "Size USD":        "size_usd",
        "Side":            "side",
        "Start Position":  "start_position",
        "Direction":       "direction",
        "Closed PnL":      "closed_pnl",
        "Fee":             "fee",
        "Crossed":         "crossed",
        "Order ID":        "order_id",
        "Trade ID":        "trade_id",
        "Transaction Hash":"tx_hash",
    })

    # is_close: only closing trades have realized PnL
    ht["is_close"] = ht["direction"].str.startswith("Close")

    # net_pnl: PnL minus fee (fee is always positive)
    ht["net_pnl"] = ht["closed_pnl"] - ht["fee"]

    # long/short label from direction
    ht["is_long"] = ht["direction"].str.contains("Long", na=False)
    ht["is_short"] = ht["direction"].str.contains("Short", na=False)

    return ht

# 2. MERGE  (trade-level join on date)

def merge_datasets(ht, fg):
    """Left-join trades onto sentiment by calendar date."""
    fg_slim = fg[["date", "value", "classification", "sentiment_binary"]].copy()
    merged = ht.merge(fg_slim, on="date", how="left")

    missing_sentiment = merged["classification"].isnull().sum()
    if missing_sentiment:
        print(f"[data_prep] {missing_sentiment} trade rows have no matching sentiment date — dropping.")
        merged = merged.dropna(subset=["classification"]).reset_index(drop=True)

    return merged

# 3. SAVE

def save_processed(merged, fg, ht):
    merged.to_csv(PROCESSED / "trades_with_sentiment.csv", index=False)
    fg.to_csv(PROCESSED / "fear_greed_clean.csv", index=False)
    ht.to_csv(PROCESSED / "trades_clean.csv", index=False)
    print(f"[data_prep] Saved processed files to {PROCESSED}")


# 4. QUICK AUDIT 

def data_audit(fg, ht, merged):
    """Return a dict of audit stats for the notebook."""
    return {
        "fg_shape": fg.shape,
        "ht_shape": ht.shape,
        "merged_shape": merged.shape,
        "fg_missing": fg.isnull().sum().to_dict(),
        "ht_missing": ht.isnull().sum().to_dict(),
        "fg_dupes": int(fg.duplicated().sum()),
        "ht_dupes": int(ht.duplicated().sum()),
        "ht_date_range": (str(ht["date"].min()), str(ht["date"].max())),
        "fg_date_range": (str(fg["date"].min()), str(fg["date"].max())),
        "unique_accounts": int(ht["account"].nunique()),
        "unique_coins": int(ht["coin"].nunique()),
        "sentiment_dist": fg["classification"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    fg = load_fear_greed()
    ht = load_trades()
    merged = merge_datasets(ht, fg)
    save_processed(merged, fg, ht)
    audit = data_audit(fg, ht, merged)
    for k, v in audit.items():
        print(f"  {k}: {v}")