# Project Workspace

Two projects in this repository. Both standalone, no shared dependencies.

## 1. Creative Writing Workshop (`creative_workshop.py`)

Standalone AI writing assistant with browser UI. Run `python3 creative_workshop.py` — it handles Ollama, model downloads, and opens a web interface automatically.

**Models:** Mistral Small 3.2 (24B creative), Qwen 3.5:4b (structural), nomic-embed-text (semantic search)

**Tools:** spaCy NER, LanguageTool proofreading, textstat readability, embedding-based context loading

See the [creative workshop docs](creative_workshop.py) header for full details.

## 2. Quant Pipeline (`quant_pipeline/`)

XGBoost trading system for SPY with triple barrier labeling, meta-labeling, and purged walk-forward CV.

**Current status:** 73.6% precision on real S&P 500 holdout data. See [`quant_pipeline/STATUS.md`](quant_pipeline/STATUS.md) for full results and roadmap to 97%.

**To run:** `cd quant_pipeline && python run_local.py`

## Session Notes (May 30, 2026)

### What was built today:
1. Creative Writing Workshop v2.0 — complete rewrite with Mistral 24B, embedding search, 4 new analysis tools, standalone launcher
2. Quant pipeline — 8-module system built from improved Gemini spec, tested on real S&P 500 data
3. Both pushed to this repo

### Quant pipeline next steps (continue from home):
- Run `python run_local.py` to fetch full dataset (SPY + 16 ETFs + VIX + FRED) via yfinance
- Re-run ensemble with full feature set (currently only OHLCV, need cross-asset + macro)
- Add alternative data (options flow, sentiment) for the push to 97%
- See `quant_pipeline/STATUS.md` for detailed roadmap

### Important:
**Revoke all GitHub tokens** shared during this conversation. Go to GitHub → Settings → Developer settings → Personal access tokens and delete them immediately.
