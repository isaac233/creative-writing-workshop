"""
Feature Engineering — Stationary transformations for XGBoost.

Rules enforced:
  1. No raw prices, volumes, or non-stationary levels.
  2. All prices → log returns or fractional differentiation.
  3. All volumes → rolling z-scores.
  4. Technical indicators → relative/stationary ratios.
  5. All features lagged by 1 day (no look-ahead).
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config import FeatureConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def log_returns(series: pd.Series) -> pd.Series:
    """Compute log returns. Stationary by construction."""
    return np.log(series / series.shift(1))


def rolling_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score normalization. Stationary."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def rolling_pct_rank(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank. Maps to [0, 1], stationary."""
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FRACTIONAL DIFFERENTIATION
# ═══════════════════════════════════════════════════════════════════════════════

def _frac_diff_weights(d: float, threshold: float = 0.01) -> np.ndarray:
    """Compute fractional differentiation weights until they drop below threshold."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])


def frac_diff(series: pd.Series, d: float = 0.4, threshold: float = 0.01) -> pd.Series:
    """
    Apply fractional differentiation to a price series.
    Preserves memory while achieving stationarity.
    From: Advances in Financial Machine Learning (de Prado, Ch. 5).
    """
    weights = _frac_diff_weights(d, threshold)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=float)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        if len(window) == width:
            result.iloc[i] = np.dot(weights, window)

    return result


def find_min_frac_diff_d(
    series: pd.Series,
    d_range: np.ndarray = np.arange(0.0, 1.05, 0.05),
    pvalue: float = 0.05,
    threshold: float = 0.01,
) -> float:
    """
    Find minimum d that achieves stationarity (ADF test).
    Preserves maximum memory while making the series stationary.
    """
    for d in d_range:
        if d == 0:
            continue
        fd = frac_diff(series.dropna(), d=d, threshold=threshold).dropna()
        if len(fd) < 50:
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(fd, maxlag=1, regression="c")
            if adf_p < pvalue:
                return round(d, 2)
        except Exception:
            continue
    return 1.0  # fallback to full differentiation


# ═══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS (stationary versions)
# ═══════════════════════════════════════════════════════════════════════════════

def sma_ratio(close: pd.Series, short: int, long: int) -> pd.Series:
    """SMA crossover ratio: SMA_short / SMA_long - 1. Centered around 0."""
    return close.rolling(short).mean() / close.rolling(long).mean() - 1


def ema_ratio(close: pd.Series, short: int, long: int) -> pd.Series:
    """EMA crossover ratio: EMA_short / EMA_long - 1."""
    return close.ewm(span=short).mean() / close.ewm(span=long).mean() - 1


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI, already bounded [0, 100]. Center to [-50, 50] for symmetry."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)) - 50  # centered at 0


def macd_signal_distance(close: pd.Series) -> pd.Series:
    """MACD histogram as fraction of price. Stationary."""
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    return (macd_line - signal_line) / close


def bollinger_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Price position within Bollinger Bands as z-score. Stationary."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - sma) / std.replace(0, np.nan)


def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR as fraction of price. Stationary volatility measure."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close


def volume_zscore(volume: pd.Series, window: int = 63) -> pd.Series:
    """Volume rolling z-score. Stationary."""
    return rolling_zscore(volume, window)


def volume_ratio(volume: pd.Series, short: int = 5, long: int = 63) -> pd.Series:
    """Short-term vs long-term volume ratio. Stationary."""
    return volume.rolling(short).mean() / volume.rolling(long).mean() - 1


# ═══════════════════════════════════════════════════════════════════════════════
#  MOMENTUM & MEAN-REVERSION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Sharpe ratio. Measures risk-adjusted momentum."""
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    return (mu / sigma.replace(0, np.nan)) * np.sqrt(252)


def drawdown_from_peak(close: pd.Series, window: int = 252) -> pd.Series:
    """Rolling drawdown from peak. Bounded [−1, 0]."""
    rolling_max = close.rolling(window, min_periods=1).max()
    return close / rolling_max - 1


def return_dispersion(etf_returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """Cross-sectional dispersion of sector returns. Higher = less correlation."""
    return etf_returns.rolling(window).std().mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDER — assembles all features from raw data
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(raw: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Build the full stationary feature matrix from raw merged data.
    All features are lagged by 1 day to prevent look-ahead bias.
    """
    cfg = cfg or FeatureConfig()
    features = pd.DataFrame(index=raw.index)

    close = raw["close"]
    high = raw["high"]
    low = raw["low"]
    volume = raw["volume"]
    ret = log_returns(close)

    print(f"  Building features from {len(raw.columns)} raw columns...")

    # ── Price-derived (stationary) ────────────────────────────────────────────
    features["log_return"] = ret

    for w in cfg.short_windows:
        features[f"return_{w}d"] = close.pct_change(w)
        features[f"rolling_sharpe_{w}d"] = rolling_sharpe(ret, w)

    for short, long in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        features[f"sma_ratio_{short}_{long}"] = sma_ratio(close, short, long)
        features[f"ema_ratio_{short}_{long}"] = ema_ratio(close, short, long)

    # ── Technical indicators (stationary) ─────────────────────────────────────
    for period in [7, 14, 21]:
        features[f"rsi_{period}"] = rsi(close, period)

    features["macd_signal_dist"] = macd_signal_distance(close)

    for w in [10, 20, 50]:
        features[f"boll_zscore_{w}"] = bollinger_zscore(close, w)

    for period in [7, 14, 21]:
        features[f"atr_ratio_{period}"] = atr_ratio(high, low, close, period)

    # ── Volume (stationary) ───────────────────────────────────────────────────
    features["vol_zscore_20"] = volume_zscore(volume, 20)
    features["vol_zscore_63"] = volume_zscore(volume, 63)
    features["vol_ratio_5_63"] = volume_ratio(volume, 5, 63)

    # ── Volatility ────────────────────────────────────────────────────────────
    for w in [10, 20, 63]:
        features[f"realized_vol_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

    features["vol_of_vol_20"] = (
        (ret.rolling(20).std() * np.sqrt(252))
        .rolling(20).std()
    )

    # ── Drawdown ──────────────────────────────────────────────────────────────
    features["drawdown_252"] = drawdown_from_peak(close, 252)

    # ── Sector ETF relative strength ──────────────────────────────────────────
    etf_cols = [c for c in raw.columns if c.startswith("etf_")]
    if etf_cols:
        etf_returns = raw[etf_cols].pct_change()
        for col in etf_cols:
            name = col.replace("etf_", "")
            # Relative return vs SPY
            features[f"rel_{name}_5d"] = (
                raw[col].pct_change(5) - close.pct_change(5)
            )
            features[f"rel_{name}_20d"] = (
                raw[col].pct_change(20) - close.pct_change(20)
            )

        features["sector_dispersion_20"] = return_dispersion(etf_returns, 20)

    # ── FRED economic indicators (stationary) ─────────────────────────────────
    fred_cols = [c for c in raw.columns if c.startswith("fred_")]
    for col in fred_cols:
        name = col.replace("fred_", "")
        series = raw[col]
        # Diff for level series, z-score for already-rate series
        features[f"{name}_diff"] = series.diff()
        features[f"{name}_zscore"] = rolling_zscore(series, cfg.zscore_window)
        features[f"{name}_mom_20"] = series.pct_change(20)

    # ── Calendar features ─────────────────────────────────────────────────────
    features["day_of_week"] = raw.index.dayofweek / 4.0  # normalized [0, 1]
    features["month"] = np.sin(2 * np.pi * raw.index.month / 12)  # cyclical

    # ══════════════════════════════════════════════════════════════════════════
    #  CRITICAL: Lag all features by 1 day to prevent look-ahead bias
    # ══════════════════════════════════════════════════════════════════════════
    features = features.shift(1)

    # Drop warmup rows
    features = features.iloc[200:]  # longest lookback is 200

    # Drop any all-NaN columns
    features = features.dropna(axis=1, how="all")

    # Fill remaining NaN with 0 (rare, from FRED gaps)
    features = features.fillna(0)

    # Replace infinities
    features = features.replace([np.inf, -np.inf], 0)

    print(f"  ✓ {len(features.columns)} features built, {len(features)} rows")
    return features


if __name__ == "__main__":
    from data_pipeline import load_parquet
    raw = load_parquet("raw_merged")
    features = build_features(raw)
    print(f"\nFeature matrix: {features.shape}")
    print(features.describe().T[["mean", "std", "min", "max"]].head(20))
