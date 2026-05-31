# Project Workspace

## Quant Pipeline — 97% Holdout-Acceptance Target

**Status:** Architecture complete. Needs full data to hit target.

**Run this at home:**
```bash
cd quant_pipeline
pip install -r requirements.txt
python run_conformal.py
```

This fetches real data from 6 sources (SPY, VIX, 18 ETFs, 13 FRED macro series, Google Trends sentiment), builds 200+ stationary features, trains a 25-model XGBoost ensemble, calibrates conformal prediction, and scans for filter combinations that achieve 97% holdout-acceptance precision.

**What's in the pipeline:**
- `run_conformal.py` — one-click full pipeline (START HERE)
- `conformal.py` — conformal prediction with selective abstention (the key innovation)
- `ROADMAP.md` — complete technical roadmap
- `data_pipeline.py`, `features.py`, `labels.py`, `regime.py`, `modeling.py`, `backtest.py` — modular components

**Key insight:** Conformal prediction provides mathematical (not empirical) precision guarantees. The model only issues a signal when the conformal prediction set is a singleton at 97% coverage — meaning the model is so confident that only one outcome is plausible. On all other days, it abstains. The tradeoff is fewer signals, but every accepted signal carries the guarantee.

**Why it needs your machine:** This container can't reach yfinance/FRED APIs. With only S&P 500 price data (34 features), the ensemble can't discriminate well enough for 97%. With the full dataset (200+ features from 6 independent sources), conformal prediction sets become tighter and more signals pass the singleton filter.

See `quant_pipeline/ROADMAP.md` for the full technical architecture.

---

## Creative Writing Workshop

```bash
python3 creative_workshop.py
```

Standalone AI writing assistant. Auto-manages Ollama, pulls models, opens browser UI.

---

**IMPORTANT: Revoke all GitHub tokens shared in the Claude conversation.**
