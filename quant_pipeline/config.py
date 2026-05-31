"""
Configuration — Centralized settings for the quant pipeline.
"""
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DUCKDB_PATH = DATA_DIR / "quant.duckdb"

for d in [DATA_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Data Settings ────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    ticker: str = "SPY"
    benchmark: str = "SPY"
    start_date: str = "2005-01-01"
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    interval: str = "1d"

    # Real alternative data sources (FRED series IDs)
    fred_series: list[str] = field(default_factory=lambda: [
        "VIXCLS",          # VIX
        "DFF",             # Fed Funds Rate
        "T10Y2Y",          # 10Y-2Y spread (yield curve)
        "T10YIE",          # 10Y breakeven inflation
        "BAMLH0A0HYM2",   # High yield spread
        "DTWEXBGS",        # Trade-weighted USD
        "DCOILWTICO",      # WTI Crude
        "GOLDAMGBD228NLBM",# Gold price
        "ICSA",            # Initial jobless claims
        "UMCSENT",         # Consumer sentiment
    ])

    # Sector ETFs for relative strength
    sector_etfs: list[str] = field(default_factory=lambda: [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
        "TLT", "HYG", "GLD", "EEM", "IWM", "QQQ",
    ])


# ─── Feature Engineering Settings ─────────────────────────────────────────────

@dataclass
class FeatureConfig:
    # Rolling windows
    short_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    long_windows: list[int] = field(default_factory=lambda: [50, 100, 200])
    all_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Fractional differentiation
    frac_diff_d: float = 0.4     # default d; pipeline finds optimal per feature
    frac_diff_threshold: float = 0.01  # weight cutoff for frac-diff kernel

    # Rolling z-score window for volume / alt data
    zscore_window: int = 63  # ~3 months

    # Stationarity test significance
    adf_pvalue: float = 0.05


# ─── Label Settings ───────────────────────────────────────────────────────────

@dataclass
class LabelConfig:
    # Triple barrier
    profit_taking: float = 0.02   # 2% take profit
    stop_loss: float = 0.01       # 1% stop loss
    max_holding_days: int = 10    # vertical barrier

    # Use daily volatility to scale barriers
    vol_lookback: int = 20
    use_dynamic_barriers: bool = True  # scale by rolling vol

    # Sample weights
    compute_sample_weights: bool = True


# ─── Model Settings ──────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Walk-forward CV
    n_splits: int = 5
    train_window: int = 504    # ~2 years of trading days
    test_window: int = 63      # ~3 months
    purge_gap: int = 10        # days to purge between train/test
    embargo_pct: float = 0.01  # fraction of train to embargo

    # XGBoost
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    })

    # Feature pruning
    max_features: int = 100  # keep top N by SHAP importance
    shap_sample_size: int = 500  # samples for SHAP computation


# ─── Backtest Settings ────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000
    commission_pct: float = 0.001     # 10bps round trip
    slippage_pct: float = 0.0005      # 5bps slippage
    min_confidence: float = 0.55      # meta-label threshold to enter
    max_position_pct: float = 1.0     # max allocation (1.0 = 100%)
    risk_free_rate: float = 0.05      # for Sharpe calculation
