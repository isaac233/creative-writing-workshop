"""
Regime Detection — Classify market regimes for conditional modeling.

Approaches:
  1. Volatility regime: rolling vol percentile → low/medium/high
  2. Trend regime: SMA slope → trending up / sideways / trending down
  3. Combined regime feature for XGBoost
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  VOLATILITY REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def volatility_regime(
    close: pd.Series,
    vol_window: int = 20,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Classify volatility regime based on rolling vol percentile.

    Returns DataFrame with:
      - realized_vol: annualized rolling volatility
      - vol_percentile: percentile rank within lookback window
      - vol_regime: 0=low, 1=medium, 2=high
    """
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    # Rolling percentile rank
    vol_pct = realized_vol.rolling(lookback, min_periods=63).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Classify: bottom 25% = low, top 25% = high, middle = medium
    regime = pd.Series(1, index=close.index, name="vol_regime", dtype=int)
    regime[vol_pct <= 0.25] = 0  # low vol
    regime[vol_pct >= 0.75] = 2  # high vol

    result = pd.DataFrame({
        "realized_vol": realized_vol,
        "vol_percentile": vol_pct,
        "vol_regime": regime,
    }, index=close.index)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  TREND REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def trend_regime(
    close: pd.Series,
    sma_window: int = 50,
    slope_window: int = 20,
    slope_threshold: float = 0.0002,
) -> pd.DataFrame:
    """
    Classify trend regime based on SMA slope.

    Returns DataFrame with:
      - sma_slope: normalized slope of SMA
      - price_vs_sma: price position relative to SMA
      - trend_regime: 0=down, 1=sideways, 2=up
    """
    sma = close.rolling(sma_window).mean()

    # Slope of SMA (change per day, normalized by price)
    sma_slope = sma.diff(slope_window) / (sma * slope_window)

    # Price vs SMA
    price_vs_sma = close / sma - 1

    # Classify
    regime = pd.Series(1, index=close.index, name="trend_regime", dtype=int)
    regime[sma_slope < -slope_threshold] = 0  # downtrend
    regime[sma_slope > slope_threshold] = 2   # uptrend

    result = pd.DataFrame({
        "sma_slope": sma_slope,
        "price_vs_sma": price_vs_sma,
        "trend_regime": regime,
    }, index=close.index)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def combined_regime(close: pd.Series) -> pd.DataFrame:
    """
    Build combined regime features:
      - vol regime (low/med/high)
      - trend regime (down/sideways/up)
      - combined regime (3x3 = 9 states, encoded as single int)
    """
    vol = volatility_regime(close)
    trend = trend_regime(close)

    result = pd.DataFrame({
        "vol_regime": vol["vol_regime"],
        "vol_percentile": vol["vol_percentile"],
        "trend_regime": trend["trend_regime"],
        "sma_slope": trend["sma_slope"],
        "price_vs_sma": trend["price_vs_sma"],
    }, index=close.index)

    # Combined: 3 * vol_regime + trend_regime → 0..8
    result["regime_combined"] = (
        result["vol_regime"] * 3 + result["trend_regime"]
    )

    return result


def build_regime_features(close: pd.Series) -> pd.DataFrame:
    """
    Build regime features for the modeling pipeline.
    All features are already stationary or categorical.
    """
    print(f"  ── Regime Detection ──")
    regimes = combined_regime(close)

    dist = regimes["regime_combined"].value_counts().sort_index()
    regime_names = {
        0: "low-vol/down", 1: "low-vol/flat", 2: "low-vol/up",
        3: "med-vol/down", 4: "med-vol/flat", 5: "med-vol/up",
        6: "high-vol/down", 7: "high-vol/flat", 8: "high-vol/up",
    }
    print(f"  Regime distribution:")
    for code, count in dist.items():
        name = regime_names.get(code, f"regime_{code}")
        pct = count / len(regimes) * 100
        print(f"    {name}: {count} days ({pct:.1f}%)")

    return regimes


if __name__ == "__main__":
    from data_pipeline import load_parquet
    raw = load_parquet("raw_merged")
    regimes = build_regime_features(raw["close"])
    print(f"\nRegime matrix: {regimes.shape}")
