"""
Trader clustering (behavioural archetypes) + next-day profitability prediction.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

CHARTS = Path(__file__).parent.parent / "output" / "charts"
TABLES = Path(__file__).parent.parent / "output" / "tables"
CHARTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


#  CLUSTERING  (behavioral archetypes)

CLUSTER_FEATURES = [
    "win_rate", "trades_per_day", "avg_size_usd",
    "long_bias", "pnl_volatility", "net_pnl", "total_fees"
]

SEGMENT_NAMES = {
    0: "Cautious Scalpers",
    1: "High-Volume Traders",
    2: "Swing / Position Traders",
    3: "Aggressive Momentum",
}


def cluster_traders(trader_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means clustering on trader behavioral features.
    Returns trader_df with 'cluster' and 'segment' columns added.
    """
    df = trader_df.copy()
    features = [f for f in CLUSTER_FEATURES if f in df.columns]

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Label clusters by net_pnl mean (descending = cluster 0 best)
    cluster_pnl = df.groupby("cluster")["net_pnl"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_pnl.index)}
    df["cluster"] = df["cluster"].map(rank_map)
    df["segment"] = df["cluster"].map(SEGMENT_NAMES)

    # PCA (Principal Component Analysis) for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    _plot_clusters(df, features, pca)
    return df


def _plot_clusters(df: pd.DataFrame, features: list, pca) -> None:
    colors = ["#E63946", "#F4A261", "#2A9D8F", "#264653"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trader Behavioral Archetypes (K-Means, k=4)", fontsize=13, fontweight="bold")

    # PCA scatter
    for cluster, group in df.groupby("cluster"):
        axes[0].scatter(group["pca1"], group["pca2"],
                        label=SEGMENT_NAMES.get(cluster, f"Cluster {cluster}"),
                        color=colors[cluster % len(colors)], alpha=0.7, s=60, edgecolors="k", linewidths=0.3)
    axes[0].set_title("PCA Projection of Trader Clusters")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    axes[0].legend(fontsize=8)

    # Feature means heatmap
    feature_means = df.groupby("cluster")[features].mean()
    feature_means.index = [SEGMENT_NAMES.get(i, f"C{i}") for i in feature_means.index]
    # Normalize each feature to [0, 1] for readability
    norm = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-9)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Normalised Feature Means per Segment")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(CHARTS / "07_trader_clusters.png", dpi=150, bbox_inches="tight")


def build_segment_sentiment_df(trader_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Join cluster labels onto trade-level data, then aggregate by segment × sentiment."""
    seg_map = trader_df.set_index("account")[["cluster", "segment"]]
    merged2 = merged.merge(seg_map, on="account", how="left")
    closes2 = merged2[merged2["is_close"]].copy()

    seg_sent = (
        closes2.groupby(["segment", "sentiment_binary"])
        .agg(
            total_pnl = ("closed_pnl", "sum"),
            win_rate  = ("closed_pnl", lambda x: (x > 0).mean()),
            n_trades  = ("closed_pnl", "count"),
            avg_size  = ("size_usd",   "mean"),
        )
        .reset_index()
    )
    seg_sent.to_csv(TABLES / "segment_by_sentiment.csv", index=False)
    return seg_sent

#  PREDICTIVE MODEL  (next-day profitability)

def build_prediction_dataset(daily_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Building a daily-level feature set for predicting next-day profitable (PnL > 0) or not.
    Features: lagged PnL, win rate, trade volume, sentiment encoding.
    Target: next-day total_pnl > 0 -> binary.
    """
    df = daily_df.sort_values("date").copy()

    # Sentiment encoding
    sent_map = {"Fear": 0, "Neutral": 1, "Greed": 2}
    df["sentiment_enc"] = df["sentiment_binary"].map(sent_map).fillna(1)

    # Features
    df["pnl_lag1"]       = df["total_pnl"].shift(1)
    df["pnl_lag2"]       = df["total_pnl"].shift(2)
    df["pnl_roll3"]      = df["total_pnl"].shift(1).rolling(3).mean()
    df["winrate_lag1"]   = df["win_rate"].shift(1)
    df["trades_lag1"]    = df["total_trades"].shift(1)
    df["size_lag1"]      = df["avg_size_usd"].shift(1)
    df["ls_ratio_lag1"]  = df["long_short_ratio"].shift(1).fillna(1)
    df["sentiment_lag1"] = df["sentiment_enc"].shift(1)

    # Target: next-day PnL > 0
    df["target"] = (df["total_pnl"].shift(-1) > 0).astype(int)

    feature_cols = [
        "pnl_lag1", "pnl_lag2", "pnl_roll3",
        "winrate_lag1", "trades_lag1", "size_lag1",
        "ls_ratio_lag1", "sentiment_enc", "sentiment_lag1"
    ]
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    df.to_csv(TABLES / "prediction_dataset.csv", index=False)
    return df, feature_cols


def train_predict_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Random Forest + GBM ensemble to predict next-day profitable flag.
    Useing 5-fold cross-validation. Returns metrics + feature importance.
    """
    X = df[feature_cols].values
    y = df["target"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)

    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")

    # Fit on full data for feature importance
    rf.fit(X, y)
    gb.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf.feature_importances_,
        "gb_importance": gb.feature_importances_,
    }).sort_values("rf_importance", ascending=False)

    importance_df.to_csv(TABLES / "feature_importance.csv", index=False)
    _plot_model(rf_scores, gb_scores, importance_df, feature_cols)

    return {
        "rf_auc_mean": rf_scores.mean(),
        "rf_auc_std":  rf_scores.std(),
        "gb_auc_mean": gb_scores.mean(),
        "gb_auc_std":  gb_scores.std(),
        "importance":  importance_df,
    }


def _plot_model(rf_scores, gb_scores, importance_df, feature_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Next-Day Profitability Prediction Model", fontsize=13, fontweight="bold")

    # CV score comparison
    labels = ["Random Forest", "Gradient Boosting"]
    means  = [rf_scores.mean(), gb_scores.mean()]
    stds   = [rf_scores.std(),  gb_scores.std()]
    bars = axes[0].bar(labels, means, yerr=stds, color=["#2A9D8F", "#264653"],
                       alpha=0.8, capsize=8, edgecolor="black", linewidth=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")
    axes[0].set_title("5-Fold CV ROC-AUC")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend()
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{mean:.3f}", ha="center", fontsize=10)

    # Feature importance
    top_n = importance_df.head(8)
    axes[1].barh(top_n["feature"][::-1], top_n["rf_importance"][::-1],
                 color="#E63946", alpha=0.8, edgecolor="black", linewidth=0.5, label="RF")
    axes[1].barh(top_n["feature"][::-1], -top_n["gb_importance"][::-1],
                 color="#F4A261", alpha=0.8, edgecolor="black", linewidth=0.5, label="GBM")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Feature Importance (RF → right, GBM ← left)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(CHARTS / "08_prediction_model.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_prep import load_fear_greed, load_trades, merge_datasets
    from metrics import daily_metrics, trader_metrics

    fg  = load_fear_greed()
    ht  = load_trades()
    merged = merge_datasets(ht, fg)
    dm  = daily_metrics(merged)
    tm  = trader_metrics(merged)

    tm_clustered = cluster_traders(tm)
    print(tm_clustered.groupby("segment")[["win_rate", "net_pnl", "trades_per_day"]].mean())

    df_pred, feat_cols = build_prediction_dataset(dm, merged)
    results = train_predict_model(df_pred, feat_cols)
    print(f"\nRF AUC: {results['rf_auc_mean']:.3f} ± {results['rf_auc_std']:.3f}")
    print(f"GB AUC: {results['gb_auc_mean']:.3f} ± {results['gb_auc_std']:.3f}")