# Assignment Write-Up
## Trader Performance vs Market Sentiment — Primetrade.ai

---

## Methodology

**Data:** Bitcoin Fear/Greed Index (2,644 daily rows, 2018–2025) merged with Hyperliquid trade records (211,224 rows, 32 unique accounts, 246 coins, May 2023 – May 2025). The 5-class sentiment was collapsed to a 3-class binary (Fear / Neutral / Greed) for most analyses. Only closing trades (Direction starts with "Close") were used for PnL and win rate calculations; opening trades contributed to volume metrics.

**Feature Engineering:** Daily aggregates (total PnL, win rate, trade count, long/short ratio, avg size) and per-trader lifetime metrics (win rate, net PnL, trades/day, long bias, PnL volatility). Sentiment was encoded ordinally (Fear=0, Neutral=1, Greed=2) for the predictive model.

**Segmentation:** K-Means clustering (k=4) on 7 trader behavioral features (win rate, trades/day, avg size USD, long bias, PnL volatility, net PnL, total fees), scaled with StandardScaler. Clusters were labeled by descending net PnL.

**Predictive Model:** Random Forest and Gradient Boosting classifiers trained on lagged daily features to predict next-day profitability (binary: total PnL > 0). Evaluated with 5-fold stratified cross-validation (ROC-AUC).

---

## Key Insights

**1. Fear days produce nearly 2× the PnL of Greed days.**  
Mean closed PnL per trade: Fear = $118.28, Greed = $59.66. Win rate: Fear = 86.3%, Greed = 80.8%. The "buy the fear" effect is empirically strong — Fear days account for $4.24M of the total realized PnL pool despite covering fewer calendar days in the overlap period.

**2. Traders make large, long-biased bets on Fear days, then flip short on Greed.**  
Long ratio: 58% on Fear vs 33% on Greed. Trade frequency spikes to ~793 trades/day on Fear vs ~294 on Greed. Average trade size is 57% larger on Fear days ($7,182 vs $4,574). The data confirms an active sentiment-conditional repositioning strategy across the trader pool.

**3. Segment behavior differs sharply: Cautious Scalpers collapse on Greed; High-Volume Traders are regime-agnostic.**  
Cautious Scalpers (large size, low long bias, selective): win rate drops from 89% on Fear to 38.8% on Greed — their edge is specific to oversold, mean-reversion conditions. High-Volume Traders maintain 84–91% win rate across all regimes, indicating their edge is structural (order flow, execution quality) rather than sentiment-conditional.

---

## Strategy Recommendations

**Strategy 1 — Sentiment-Conditional Position Sizing**  
For Swing/Position Traders and High-Volume Traders: scale position size to 1.25× on Fear days and reduce to 0.75× on Greed days. For Cautious Scalpers: reduce to 0.50× or sit out entirely on Greed days (win rate halves). Rationale: the 2× PnL differential on Fear days justifies larger exposure for the segments whose edge holds across regimes; protecting against Cautious Scalper drawdowns on Greed is equally important.

**Strategy 2 — Sentiment-Conditional Directional Tilt**  
Formalize the observed natural bias as a trading rule: target ≥ 60% long entries on Fear days, ≥ 50% short entries on Greed days (segment-adjusted). Specifically, Aggressive Momentum traders (currently 25% long bias) should suppress their short inclination on Fear days — the long trade win rate in this segment is ~22pp higher than short on Fear.

---

## Model Result

5-fold CV ROC-AUC: RF = 0.677 ± 0.129, GBM = 0.666 ± 0.113. Top predictors: rolling 3-day PnL, lagged win rate, prior trade size — behavioral momentum dominates. Sentiment features ranked last (~1–2% importance), confirming that market regime is a *context* signal, not a direct PnL driver.