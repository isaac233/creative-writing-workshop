# PROJECT PLAN — 97% Holdout-Acceptance XGBoost Pipeline
# Last Updated: 2026-05-31 by Claude Opus 4.6
# Status: IN PROGRESS

## ═══════════════════════════════════════════════════════════════
## PROJECT GOAL
## ═══════════════════════════════════════════════════════════════
##
## Build an XGBoost-based trading model for SPY that achieves a
## 97% holdout-acceptance probability score on 2-week (10 trading day)
## forward predictions.
##
## "Holdout-acceptance probability" = precision on ACCEPTED signals only.
## The model uses conformal prediction with selective abstention:
## it only issues a trade signal when it is mathematically certain
## (at the 97% coverage level) that the prediction is correct.
## On all other days, it abstains.
##
## The model is LONG-ONLY. No short selling. No margin.
##
## ═══════════════════════════════════════════════════════════════

## ═══════════════════════════════════════════════════════════════
## REPO LOCATION
## ═══════════════════════════════════════════════════════════════
##
## GitHub: https://github.com/isaac233/creative-writing-workshop
## Directory: quant_pipeline/
## Main entry point: quant_pipeline/run_conformal.py
##
## To run: cd quant_pipeline && python run_conformal.py
##
## ═══════════════════════════════════════════════════════════════

## ═══════════════════════════════════════════════════════════════
## ARCHITECTURE OVERVIEW
## ═══════════════════════════════════════════════════════════════
##
## Layer 1: Data ingestion (multi-source) → raw parquet
## Layer 2: Feature engineering (all stationary, all lagged by 1 day)
## Layer 3: XGBoost ensemble (25 diverse models)
## Layer 4: Conformal prediction (MAPIE) with selective abstention
## Layer 5: Meta-label position sizing
## Layer 6: Backtest with transaction costs
##
## Key innovation: Conformal prediction provides MATHEMATICAL
## (not empirical) precision guarantees via selective abstention.
## The model only trades when the conformal prediction set is a
## singleton at 97% coverage — meaning only one outcome is plausible.
##
## ═══════════════════════════════════════════════════════════════

## ═══════════════════════════════════════════════════════════════
## FILE INVENTORY
## ═══════════════════════════════════════════════════════════════
##
## quant_pipeline/
## ├── PROJECT_PLAN.md        ← THIS FILE (master plan, always update)
## ├── run_conformal.py       ← main entry point (fetches data + runs all)
## ├── conformal.py           ← conformal prediction + selective abstention
## ├── config.py              ← centralized settings
## ├── data_pipeline.py       ← data ingestion + storage
## ├── features.py            ← stationarity transforms
## ├── labels.py              ← triple barrier + sample weights
## ├── regime.py              ← market regime detection
## ├── modeling.py            ← XGBoost + CV + SHAP + baselines
## ├── backtest.py            ← walk-forward simulation
## ├── requirements.txt       ← Python dependencies
## ├── ROADMAP.md             ← technical architecture doc
## ├── STATUS.md              ← session status
## └── data/parquet/           ← stored data (gitignored except seed data)
##
## ═══════════════════════════════════════════════════════════════


## ═══════════════════════════════════════════════════════════════
## CURRENT RESULTS (update after each run)
## ═══════════════════════════════════════════════════════════════
##
## Date: 2026-05-31
## Data: Real S&P 500 OHLCV 2010-2018 (2259 days, 34 features)
## Best holdout precision: 73.6% (ensemble + regime filter)
## Conformal result: 40% precision (insufficient features for
##   conformal sets to narrow to singletons)
## Bottleneck: Not enough independent signal sources.
##   Price/volume alone cannot drive conformal sets to singleton
##   at 97% coverage. Need cross-asset, macro, sentiment, options data.
##
## ═══════════════════════════════════════════════════════════════


## ═══════════════════════════════════════════════════════════════
## PRIORITY TASK LIST
## ═══════════════════════════════════════════════════════════════
##
## Each task below is a self-contained unit of work.
## Status: ✅ DONE | 🔄 IN PROGRESS | ⬚ TODO | ❌ BLOCKED
##
## IMPORTANT: After completing each task:
##   1. Update this file's status for that task
##   2. Update CURRENT RESULTS section if metrics changed
##   3. Commit and push: git add -A && git commit -m "..." && git push
##   4. Run the pipeline and record new metrics
##
## ═══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# TASK 0: Foundation (COMPLETED)
# Status: ✅ DONE
# ─────────────────────────────────────────────────────────────
#
# What was done:
# - Built 8-module pipeline (data, features, labels, regime, modeling, backtest, conformal, config)
# - Implemented conformal prediction with MAPIE + manual fallback
# - Created run_conformal.py that fetches real data and runs full scan
# - Tested on real S&P 500 OHLCV data (2010-2018)
# - Pushed everything to GitHub repo
#
# Files modified: all quant_pipeline/*.py
# Commit: a50c249
#


# ─────────────────────────────────────────────────────────────
# TASK 1: CBOE Put/Call Ratio
# Status: ⬚ TODO
# Priority: HIGHEST — strongest free sentiment signal
# ─────────────────────────────────────────────────────────────
#
# WHY: The CBOE equity put/call ratio is a contrarian sentiment
# indicator. When it spikes above 1.0, fear is extreme and markets
# tend to rebound. When it drops below 0.6, complacency signals risk.
# It adds an INDEPENDENT signal dimension (options market sentiment)
# that price/volume cannot capture.
#
# HOW TO IMPLEMENT:
#
# Option A (preferred): FRED has the CBOE put/call ratio
#   - FRED series ID: check for "CBOE" or use pandas-datareader
#   - Already have FRED fetching in run_conformal.py
#   - Just add the series ID to the FRED fetch list
#
# Option B: Download CSV from CBOE website
#   - URL: https://www.cboe.com/us/options/market_statistics/historical_data/
#   - Files: "Cboe Total Exchange Volume and Put/Call Ratios"
#   - Parse the CSV, align to trading calendar, save to parquet
#
# Option C: Compute from SPY options via yfinance
#   - yf.Ticker("SPY").options gives expiration dates
#   - For each expiration, get puts and calls
#   - Sum put volume / sum call volume = daily put/call ratio
#   - This gives SPY-specific ratio, which is even better than CBOE total
#
# FEATURES TO CREATE (in features section of run_conformal.py):
#   - put_call_raw (the ratio itself, already somewhat stationary)
#   - put_call_z_20 (rolling 20-day z-score)
#   - put_call_z_63 (rolling 63-day z-score)
#   - put_call_5d_change (5-day change in ratio)
#   - put_call_extreme_high (binary: z > 2, extreme fear)
#   - put_call_extreme_low (binary: z < -2, extreme greed)
#
# STATIONARITY: The ratio itself is already roughly stationary.
# Z-scores and changes make it fully stationary.
#
# LAG: Must be lagged by 1 day like all other features.
#
# WHERE TO ADD CODE:
#   - In run_conformal.py, after the FRED fetch section
#   - Add new features in the "PHASE 2: Feature Engineering" section
#
# VALIDATION:
#   - After adding, check that features are not NaN for most of the dataset
#   - Check correlation with existing features (should be low, <0.3)
#   - Run pipeline and compare precision vs baseline
#
# ESTIMATED IMPACT: Medium-high. Adds contrarian sentiment dimension.
#


# ─────────────────────────────────────────────────────────────
# TASK 2: SPY Options Implied Volatility Skew
# Status: ⬚ TODO
# Priority: HIGH — smart money positioning signal
# ─────────────────────────────────────────────────────────────
#
# WHY: The IV skew (difference between OTM put IV and ATM IV)
# tells you how much institutional investors are paying for
# downside protection. When skew is steep, smart money is hedging
# hard — this front-runs VIX moves by 1-3 days. It's one of the
# most predictive options-derived features for equity returns.
#
# HOW TO IMPLEMENT:
#
# 1. Get SPY options chain via yfinance:
#    ```python
#    spy = yf.Ticker("SPY")
#    expirations = spy.options  # list of expiration dates
#    # Pick nearest expiration 20-40 DTE
#    chain = spy.option_chain(expiration)
#    puts = chain.puts
#    calls = chain.calls
#    ```
#
# 2. Compute skew:
#    - ATM strike = closest to current price
#    - OTM put strike = 5% below current price (25-delta approximation)
#    - skew = OTM_put_IV - ATM_call_IV
#    - normalized_skew = skew / ATM_call_IV
#
# 3. CAUTION: yfinance options data is current-day only.
#    For HISTORICAL skew, you need to either:
#    a) Start collecting daily going forward (build your own history)
#    b) Use a proxy: VIX / historical_vol_20d ratio approximates skew
#    c) Use CBOE SKEW index if available via FRED or yfinance (^SKEW)
#
# RECOMMENDED APPROACH for immediate use:
#   - Fetch ^SKEW from yfinance (CBOE SKEW Index, available historically)
#   - This IS the skew, already computed by CBOE
#   - yf.download("^SKEW", start="2005-01-01")
#
# FEATURES TO CREATE:
#   - skew_raw (CBOE SKEW index value, ~100-150 range)
#   - skew_z_20 (rolling 20-day z-score)
#   - skew_z_63 (rolling 63-day z-score)
#   - skew_diff5 (5-day change)
#   - skew_vs_vix (SKEW / VIX ratio — divergence signal)
#   - skew_extreme (binary: z > 2)
#
# STATIONARITY: Z-scores and diffs are stationary.
#
# WHERE TO ADD CODE:
#   - In run_conformal.py PHASE 1, after VIX fetch
#   - Add: skew = yf.download("^SKEW", start="2005-01-01")
#   - Features in PHASE 2 after VIX features section
#
# ESTIMATED IMPACT: High. Independent signal from options market.
#


# ─────────────────────────────────────────────────────────────
# TASK 3: Alpha Vantage News Sentiment
# Status: ⬚ TODO
# Priority: HIGH — NLP sentiment from news headlines
# ─────────────────────────────────────────────────────────────
#
# WHY: News sentiment captures information that price hasn't
# fully absorbed yet. Alpha Vantage provides pre-scored sentiment
# for free (25 calls/day on free tier, 500/day on $50/mo plan).
# This is the easiest way to add NLP sentiment without running
# your own model.
#
# HOW TO IMPLEMENT:
#
# 1. Get free API key: https://www.alphavantage.co/support/#api-key
#
# 2. Fetch sentiment:
#    ```python
#    import requests
#    AV_KEY = "YOUR_KEY"
#    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=SPY&apikey={AV_KEY}"
#    r = requests.get(url)
#    data = r.json()
#    # data["feed"] contains articles with sentiment scores
#    ```
#
# 3. LIMITATION: Free tier only gives recent news, not historical.
#    For historical sentiment, you need either:
#    a) Start collecting daily going forward
#    b) Use Google Trends as a proxy (already in pipeline)
#    c) Use FinBERT on scraped historical headlines (Task 5)
#
# ALTERNATIVE for historical data: Alpha Vantage also has
#   MARKET_SENTIMENT endpoint that gives aggregate market mood.
#   Check if this has historical depth.
#
# FEATURES TO CREATE:
#   - news_sentiment_avg (average sentiment score, -1 to 1)
#   - news_sentiment_z (rolling z-score)
#   - news_volume (number of articles — high volume = event)
#   - news_sentiment_momentum (5-day change in sentiment)
#
# WHERE TO ADD CODE:
#   - New section in run_conformal.py PHASE 1: "Alpha Vantage Sentiment"
#   - Store API key in environment variable: ALPHAVANTAGE_API_KEY
#   - Gracefully skip if key not set (print warning, don't crash)
#
# ESTIMATED IMPACT: Medium-high for going forward, limited for backtest.
#


# ─────────────────────────────────────────────────────────────
# TASK 4: FMP Insider Trading Aggregates
# Status: ⬚ TODO
# Priority: MEDIUM-HIGH — documented alpha source
# ─────────────────────────────────────────────────────────────
#
# WHY: When corporate insiders cluster-buy their own stock,
# it consistently predicts positive forward returns. This is
# one of the most academically documented alpha signals.
# For SPY, we aggregate insider activity across S&P 500 constituents.
#
# HOW TO IMPLEMENT:
#
# 1. Get free FMP API key: https://financialmodelingprep.com/developer
#    Free tier: 250 API calls/day
#
# 2. Fetch insider trading:
#    ```python
#    FMP_KEY = "YOUR_KEY"
#    # Get recent insider trades for major SPY constituents
#    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol=SPY&apikey={FMP_KEY}"
#    # Alternatively, get bulk insider data:
#    url = f"https://financialmodelingprep.com/api/v4/insider-trading?page=0&apikey={FMP_KEY}"
#    ```
#
# 3. Aggregate into daily signals:
#    - Count buy vs sell transactions per day across S&P 500 stocks
#    - Compute rolling buy/sell ratio (20-day window)
#    - Compute dollar-weighted buy/sell ratio
#
# ALTERNATIVE (no API key needed):
#   - SEC EDGAR Form 4 filings are public
#   - Parse from: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4
#   - More complex but free and unlimited
#
# FEATURES TO CREATE:
#   - insider_buy_count_20d (rolling 20-day count of buys)
#   - insider_sell_count_20d (rolling 20-day count of sells)
#   - insider_buy_sell_ratio (buy_count / sell_count)
#   - insider_net_z (z-score of net buying)
#   - insider_cluster_buy (binary: >3 buys in 5 days = cluster)
#
# WHERE TO ADD CODE:
#   - New section in run_conformal.py PHASE 1: "Insider Trading"
#   - Store API key in env var: FMP_API_KEY
#   - Gracefully skip if key not set
#
# ESTIMATED IMPACT: Medium-high. Well-documented alpha, independent signal.
#


# ─────────────────────────────────────────────────────────────
# TASK 5: FinBERT Local Sentiment Scoring
# Status: ⬚ TODO
# Priority: MEDIUM — requires more setup but powerful
# ─────────────────────────────────────────────────────────────
#
# WHY: Instead of relying on pre-scored sentiment from an API,
# run your own financial NLP model locally. FinBERT is a BERT
# model fine-tuned on financial text. It scores headlines as
# positive/negative/neutral with high accuracy. This gives you
# historical sentiment if you can get historical headlines.
#
# HOW TO IMPLEMENT:
#
# 1. Install:
#    pip install transformers torch
#
# 2. Load model:
#    ```python
#    from transformers import AutoModelForSequenceClassification, AutoTokenizer
#    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#    ```
#
# 3. Score headlines:
#    ```python
#    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
#    outputs = model(**inputs)
#    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#    # probs[0] = [positive, negative, neutral]
#    sentiment = probs[0][0].item() - probs[0][1].item()  # -1 to +1
#    ```
#
# 4. Get historical headlines:
#    - Alpha Vantage news endpoint (limited history)
#    - NewsAPI.org (free tier: 100 requests/day, 1 month history)
#    - Or: scrape financial RSS feeds going forward
#
# 5. Aggregate:
#    - Average daily sentiment across all SPY-related headlines
#    - Rolling z-scores for stationarity
#
# FEATURES TO CREATE:
#   - finbert_sentiment (daily average, -1 to 1)
#   - finbert_z_10 (rolling 10-day z-score)
#   - finbert_z_20 (rolling 20-day z-score)
#   - finbert_momentum (5-day sentiment change)
#   - finbert_dispersion (std of headline sentiments — disagreement signal)
#
# NOTE: This runs on CPU. ~1 second per headline. For daily batch
# processing of 20-50 headlines, total time is <1 minute.
#
# WHERE TO ADD CODE:
#   - New file: quant_pipeline/sentiment.py
#   - Called from run_conformal.py PHASE 1
#   - Gracefully skip if transformers/torch not installed
#
# ESTIMATED IMPACT: Medium. Powerful but limited by headline history.
#


# ─────────────────────────────────────────────────────────────
# TASK 6: Earnings Calendar Density
# Status: ⬚ TODO
# Priority: MEDIUM — captures volatility event timing
# ─────────────────────────────────────────────────────────────
#
# WHY: Volatility increases around earnings season. The number
# of major SPY-constituent earnings reports in the next 5-10 days
# creates a "vol event density" feature. This helps the model
# know when to abstain (high earnings density = uncertain) vs
# when to trade (low density = calmer regime).
#
# HOW TO IMPLEMENT:
#
# 1. Get earnings calendar:
#    ```python
#    import yfinance as yf
#    spy = yf.Ticker("SPY")
#    # Get top holdings
#    # For each holding, get earnings dates
#    for ticker in ["AAPL", "MSFT", "AMZN", "NVDA", "GOOG", ...]:
#        t = yf.Ticker(ticker)
#        cal = t.earnings_dates  # DataFrame with dates
#    ```
#
# 2. Count earnings in rolling window:
#    - For each trading day, count how many top-50 SPY stocks
#      report earnings in the next 5 trading days
#    - Also count how many reported in the past 5 days
#
# FEATURES TO CREATE:
#   - earnings_ahead_5d (count of top-50 SPY stocks reporting in next 5 days)
#   - earnings_behind_5d (count reported in past 5 days)
#   - earnings_density (combined)
#   - earnings_season (binary: density > threshold)
#
# CAUTION: yfinance earnings_dates may have limited history.
# If so, use a simpler proxy: January/April/July/October = earnings season.
#
# WHERE TO ADD CODE:
#   - In run_conformal.py PHASE 1, new section "Earnings Calendar"
#   - Gracefully handle missing data
#
# ESTIMATED IMPACT: Medium. Helps the model know WHEN to abstain.
#


# ─────────────────────────────────────────────────────────────
# TASK 7: CFTC Commitment of Traders (COT) Data
# Status: ⬚ TODO
# Priority: MEDIUM-LOW — weekly frequency limits granularity
# ─────────────────────────────────────────────────────────────
#
# WHY: COT reports show how commercial hedgers, large speculators,
# and small speculators are positioned in S&P 500 futures. Extreme
# positions by large speculators are contrarian signals.
#
# HOW TO IMPLEMENT:
#
# 1. Download from CFTC:
#    - URL: https://www.cftc.gov/dea/futures/deacmesf.htm
#    - Or use Python package: pip install cot-reports
#    ```python
#    from cot_reports import cot_report
#    df = cot_report(report_type='legacy_fut', cot_report_type='all_disagg')
#    # Filter for S&P 500 E-mini futures
#    ```
#
# 2. ALTERNATIVE: Some FRED series contain COT-derived data
#
# 3. Since this is WEEKLY data (released every Friday for Tuesday positions),
#    forward-fill to daily frequency
#
# FEATURES TO CREATE:
#   - cot_large_spec_net (net long/short position of large speculators)
#   - cot_large_spec_z (rolling z-score, ~1 year window)
#   - cot_commercial_net (commercial hedger positioning)
#   - cot_small_spec_net (retail positioning)
#   - cot_spec_extreme (binary: z > 2 or z < -2)
#
# WHERE TO ADD CODE:
#   - In run_conformal.py PHASE 1, new section "CFTC COT Data"
#
# ESTIMATED IMPACT: Medium-low due to weekly frequency.
#


# ─────────────────────────────────────────────────────────────
# TASK 8: Additional FRED Economic Series
# Status: ⬚ TODO
# Priority: MEDIUM — deepens macro signal
# ─────────────────────────────────────────────────────────────
#
# WHY: More macro indicators = more independent signals for regime
# detection. Each one captures a different aspect of the economy.
#
# SERIES TO ADD (append to existing FRED fetch list):
#   - "T10Y3M"    → 10Y-3M spread (most sensitive recession indicator)
#   - "BAA10Y"    → BAA-10Y spread (corporate credit risk)
#   - "TEDRATE"   → TED spread (interbank stress)
#   - "M2SL"      → M2 money supply (liquidity)
#   - "CPIAUCSL"  → CPI (inflation)
#   - "PAYEMS"    → Nonfarm payrolls (employment)
#   - "PERMIT"    → Building permits (leading indicator)
#   - "AWHMAN"    → Avg weekly hours manufacturing (leading)
#   - "NEWORDER"  → New orders index (leading)
#   - "STLFSI2"   → St. Louis Financial Stress Index
#
# For each, create:
#   - {name}_diff (first difference)
#   - {name}_z (rolling 63-day z-score)
#   - {name}_mom_20 (20-day momentum)
#
# WHERE TO ADD: In run_conformal.py, extend the fred_ids list.
# Already have the FRED fetch loop — just add series IDs.
#
# ESTIMATED IMPACT: Medium. Deepens macro regime signal.
#


# ─────────────────────────────────────────────────────────────
# TASK 9: Pipeline Integration & Testing
# Status: ⬚ TODO
# Priority: DO AFTER TASKS 1-3 ARE DONE
# ─────────────────────────────────────────────────────────────
#
# After adding new data sources:
#
# 1. Run full pipeline: python run_conformal.py
# 2. Record new metrics in CURRENT RESULTS section above
# 3. Compare feature count: target 200+ features
# 4. Check conformal singleton rate — should increase with more features
# 5. Check holdout-acceptance probability — target 97%
# 6. If still below 97%, analyze:
#    a. Which features have highest SHAP importance?
#    b. Are new features correlated with existing ones? (want low correlation)
#    c. Is the ensemble diverse enough? (check individual model disagreement)
#    d. Is the calibration set large enough? (need 200+ samples minimum)
# 7. Iterate: adjust α, add more models, try different targets
#
# TUNING KNOBS (if 97% not hit after adding data):
#   - Increase ensemble from 25 to 40 models
#   - Try regime-conditioned conformal (different α per regime)
#   - Try different target definitions:
#     a. "Positive return" (current)
#     b. "No drawdown > 0.5% in 10 days" (regime persistence)
#     c. "Return > 1% in 10 days" (strong move)
#   - Add LightGBM and CatBoost to ensemble (model class diversity)
#   - Isotonic calibration before conformal (better probability estimates)
#
# WHERE TO MODIFY: run_conformal.py PHASE 3
#


# ─────────────────────────────────────────────────────────────
# TASK 10: Backtest Integration
# Status: ⬚ TODO
# Priority: DO AFTER 97% IS ACHIEVED
# ─────────────────────────────────────────────────────────────
#
# Once 97% holdout-acceptance is achieved:
#
# 1. Connect conformal signals to backtest.py
# 2. Run walk-forward backtest with:
#    - 10bps commission + 5bps slippage per trade
#    - Position sizing by conformal confidence
#    - Long-only constraint
# 3. Compare vs buy-and-hold SPY
# 4. Report: Sharpe, Sortino, max drawdown, win rate, profit factor
# 5. Plot equity curve
#
# WHERE TO MODIFY: backtest.py, connect to conformal.py output
#


## ═══════════════════════════════════════════════════════════════
## EXECUTION INSTRUCTIONS FOR ANY AI MODEL
## ═══════════════════════════════════════════════════════════════
##
## 1. Read this file FIRST. It is the source of truth.
## 2. Check which tasks are ⬚ TODO and start with the lowest-numbered one.
## 3. All code changes go in quant_pipeline/run_conformal.py unless
##    a new file is specified in the task.
## 4. After completing a task:
##    a. Update this file (change ⬚ TODO → ✅ DONE, add notes)
##    b. Test: python run_conformal.py (if on user's machine)
##    c. Commit: git add -A && git commit -m "Task N: description" && git push
## 5. If you run out of context/tokens:
##    a. Update this file with your progress
##    b. Commit and push
##    c. The next session can read this file and continue
## 6. The GitHub token for pushing:
##    User must provide a fresh token — do NOT store tokens in files.
##    Ask user: "Please provide your GitHub PAT for pushing."
## 7. User's GitHub: isaac233
## 8. Repo: isaac233/creative-writing-workshop
## 9. Branch: main
##
## CODING STANDARDS:
##   - All features must be STATIONARY (z-scores, diffs, ratios, log returns)
##   - All features must be LAGGED by 1 day (shift(1)) — no look-ahead
##   - All new data sources must GRACEFULLY SKIP if unavailable (try/except)
##   - All API keys from environment variables (never hardcoded)
##   - Print progress messages so user can see what's happening
##   - Save intermediate results to parquet for debugging
##
## ═══════════════════════════════════════════════════════════════


## ═══════════════════════════════════════════════════════════════
## CHANGE LOG
## ═══════════════════════════════════════════════════════════════
##
## 2026-05-31 (Claude Opus 4.6, session 1):
##   - Created full pipeline: 8 Python modules
##   - Tested on real S&P 500 data (2010-2018)
##   - Implemented conformal prediction with MAPIE
##   - Best precision: 73.6% (ensemble + regime filters)
##   - Conformal framework works but needs more features
##   - Created this PROJECT_PLAN.md
##   - Pushed to GitHub: isaac233/creative-writing-workshop
##
## ═══════════════════════════════════════════════════════════════
