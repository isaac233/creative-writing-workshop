# Quantitative Pipeline — SPY Long-Only Strategy

XGBoost-based trading system with triple barrier labeling, meta-labeling, and purged walk-forward cross-validation.

## Architecture

```
main.py          → orchestrator (runs full pipeline)
config.py        → centralized settings
data_pipeline.py → market + FRED + sector ETF data → DuckDB/Parquet
features.py      → stationary transforms (frac-diff, rolling z-scores, RSI ratios)
labels.py        → triple barrier + sample weights + meta-labels
regime.py        → volatility/trend regime detection
modeling.py      → XGBoost (two-stage) + SHAP pruning + baselines
backtest.py      → walk-forward simulation with transaction costs
```

## Quick Start

```bash
cd quant_pipeline
pip install -r requirements.txt
python main.py
```

## Key Design Decisions

**Real data, not mocks.** Pipeline ingests SPY OHLCV, 16 sector/factor ETFs, and 10 FRED economic series. No synthetic variables — quality over quantity.

**Strict stationarity.** No raw prices or volumes enter the model. Everything is converted to log returns, rolling z-scores, fractional differentiation, or relative ratios.

**Triple barrier + meta-labeling.** Stage 1 predicts direction (profit/stop/time barriers). Stage 2 predicts probability that the signal is profitable. Meta-label confidence drives position sizing.

**Purged walk-forward CV.** No standard KFold. Training/test splits are temporal with purge gaps and embargo periods to prevent leakage from overlapping labels.

**Sample weights.** Overlapping triple-barrier labels are weighted by average uniqueness to prevent overfitting to correlated samples.

**Transaction costs.** Backtest applies commission (10bps) and slippage (5bps) to every trade. No unrealistic assumptions.

## Pipeline Stages

1. **Data Ingestion** — Fetch from yfinance + FRED, merge on trading calendar, store in DuckDB + Parquet
2. **Feature Engineering** — 100+ stationary features from price, volume, technicals, cross-asset, economic data
3. **Regime Detection** — Volatility percentile + SMA slope → 9 market states
4. **Labeling** — Triple barrier with dynamic vol-scaled thresholds + sample weights
5. **Modeling** — XGBoost multi-class → SHAP pruning → retrain → meta-label binary model
6. **Backtest** — Event-driven walk-forward with costs, confidence-based sizing, vs buy-and-hold benchmark
