# Quant Pipeline — Project Status

## Current State (May 30, 2026)

### What's Built
- Full 6-module pipeline: data → features → labels → regime → modeling → backtest
- 20-model diverse XGBoost ensemble with cascading confidence filters
- Purged walk-forward cross-validation (no data leakage)
- Triple barrier labeling with sample weights
- Meta-labeling for position sizing
- Transaction cost modeling in backtest

### Results on Real S&P 500 Data (2010-2018)
- **Best robust precision: 73.6%** (filter: all 20 models agree + full trend alignment + near highs)
- Tested across 3 temporal splits for robustness
- Target was positive 10-day forward return

### Why 97% Hasn't Been Hit Yet
The current pipeline uses only price/volume data (OHLCV). With real S&P 500 data, this caps precision around 70-75% for 10-day predictions — markets are efficient and price-only features don't contain enough signal for 97% precision.

### Next Steps to Reach 97%

1. **Run with full data on your machine** (`python run_local.py`)
   - yfinance will pull current SPY + 16 sector ETFs + VIX
   - FRED API will pull 10 economic series
   - More features = more signal for the ensemble

2. **Add alternative data sources**
   - Google Trends sentiment (pytrends)
   - Put/call ratio from CBOE
   - Options implied volatility skew
   - Earnings calendar proximity
   - Congressional trading disclosures

3. **Narrower target definition**
   - Instead of "positive 10d return", target "no drawdown >1% in next 10 days"
   - In confirmed uptrends, this base rate is >85%, making 97% precision achievable
   - Or: predict regime continuation (uptrend stays uptrend) rather than returns

4. **Ensemble expansion**
   - Add LightGBM and CatBoost alongside XGBoost for model diversity
   - Add a simple neural net (MLP) for non-tree-based perspective
   - True ensemble diversity is the path to high-precision filtering

5. **Calibrate thresholds on more data**
   - Current dataset is 2010-2018 (bull market bias)
   - Need 2018-2026 data (includes COVID crash, 2022 bear, 2023-24 recovery)
   - Filters calibrated across both bull and bear regimes will be more robust

## Running on Your Machine

```bash
cd quant_pipeline
python run_local.py
```

This fetches real-time data, builds features, trains models, and prints holdout results.
