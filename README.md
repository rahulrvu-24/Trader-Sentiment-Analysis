# Primetrade.ai — Data Science Intern Assignment
## Trader Performance vs Market Sentiment (Fear/Greed Index)

---

## Project Structure

```
primetrade-assignment/
├── data/
│   ├── raw/                          # Source CSVs (fear_greed_index.csv, historical_data.csv)
│   └── processed/                    # Cleaned & merged outputs
├── output/
│   ├── charts/                       # PNG charts (8 files)
│   └── tables/                       # Summary CSV tables (4 files)
├── notebooks/
│   └── analysis.ipynb                # Main notebook
├── src/
│   ├── data_prep.py                  # Loading, cleaning, merging
│   ├── metrics.py                    # Feature engineering
│   ├── analysis.py                   # Part B analysis + chart generation
│   ├── models.py                     # K-Means clustering + RF/GBM predictor
│   └── dashboard.py                  # Streamlit app
├── requirements.txt
├── README.md
└── WRITEUP.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/rahulrvu-24/primetrade-assignment.git
cd primetrade-assignment
pip install -r requirements.txt
```

### 2. Add raw data

Place the two source CSVs in `data/raw/`:
```
data/raw/fear_greed_index.csv
data/raw/historical_data.csv
```

---

## How to Run

### Option A — Jupyter Notebook (recommended)

```bash
# Generate the notebook (if not already present)
python3 generate_notebook.py

# Launch Jupyter
jupyter notebook notebooks/analysis.ipynb
```

Run all cells in order. Parts A → B → C → Predictive Model.  
Charts are saved to `output/charts/` and tables to `output/tables/` automatically.

### Option B — Run the pipeline from src/ directly

```bash
# Step 1: Data prep + clean
python3 src/data_prep.py

# Step 2: Charts & analysis
python3 src/analysis.py

# Step 3: Clustering + prediction model
python3 src/models.py
```

### Option C — Streamlit Dashboard

```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501`. Use the sidebar to filter by sentiment, accounts, and date range.

---

## Outputs

| File | Description |
|------|-------------|
| `output/charts/01_pnl_by_sentiment.png` | PnL distribution and mean PnL by Fear/Greed |
| `output/charts/02_winrate_volume_by_sentiment.png` | Daily win rate and trade volume |
| `output/charts/03_behavior_by_sentiment.png` | Trade size, long ratio, size violin |
| `output/charts/04_segment_heatmap.png` | Segment × Sentiment PnL and win rate heatmap |
| `output/charts/05_cumulative_pnl.png` | Cumulative PnL time series with sentiment shading |
| `output/charts/06_cluster_profiles.png` | Segment bar charts |
| `output/charts/07_trader_clusters.png` | PCA scatter + feature means heatmap |
| `output/charts/08_prediction_model.png` | CV ROC-AUC and feature importance |
| `output/tables/sentiment_pnl_stats.csv` | Mean/median/win rate per sentiment |
| `output/tables/segment_by_sentiment.csv` | Segment × sentiment performance |
| `output/tables/prediction_dataset.csv` | ML-ready daily feature set |
| `output/tables/feature_importance.csv` | RF + GBM feature importances |

---

## Data Notes

- **Fear/Greed Index**: 5-class (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`) collapsed to 3-class binary for analysis.
- **Trades**: `Closed PnL` is non-zero only on closing trades (`Direction` starts with "Close"). Opening trades are included in volume metrics but excluded from PnL aggregates.
- **Leverage**: No explicit leverage column in the dataset. `avg_size_usd` is used as a leverage-proxy (larger positions per account → higher effective exposure).
- **Merge**: 6 trade rows (0.003%) fell outside the Fear/Greed date coverage and were dropped.
