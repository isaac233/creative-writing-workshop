"""
Labels — Triple Barrier Method + Sample Weights + Meta-Labeling.

Implements:
  1. Triple barrier labeling (profit/stop/time barriers)
  2. Dynamic barriers scaled by rolling volatility
  3. Sample weights via average uniqueness (de Prado Ch. 4)
  4. Meta-labeling: probability that a primary signal is correct
"""
import logging

import numpy as np
import pandas as pd

from config import LabelConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DAILY VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════════

def daily_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Compute daily volatility as exponentially-weighted std of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=lookback).std()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIPLE BARRIER METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def triple_barrier_labels(
    close: pd.Series,
    cfg: LabelConfig | None = None,
) -> pd.DataFrame:
    """
    Apply the triple barrier method to generate labels.

    Returns DataFrame with columns:
      - label:    1 (profit hit), -1 (stop hit), 0 (time barrier hit)
      - ret:      actual return at barrier touch
      - barrier:  which barrier was hit ('profit', 'stop', 'time')
      - t_end:    date when barrier was touched

    Dynamic barriers: when use_dynamic_barriers=True, profit and stop
    thresholds are scaled by rolling daily volatility.
    """
    cfg = cfg or LabelConfig()

    vol = daily_volatility(close, cfg.vol_lookback) if cfg.use_dynamic_barriers else None

    labels = []

    for i in range(len(close) - cfg.max_holding_days):
        entry_date = close.index[i]
        entry_price = close.iloc[i]

        # Dynamic or fixed barriers
        if cfg.use_dynamic_barriers and vol is not None and not np.isnan(vol.iloc[i]):
            daily_vol = vol.iloc[i]
            pt_threshold = cfg.profit_taking * daily_vol * np.sqrt(cfg.max_holding_days)
            sl_threshold = cfg.stop_loss * daily_vol * np.sqrt(cfg.max_holding_days)
            # Floor to avoid tiny barriers
            pt_threshold = max(pt_threshold, 0.005)
            sl_threshold = max(sl_threshold, 0.003)
        else:
            pt_threshold = cfg.profit_taking
            sl_threshold = cfg.stop_loss

        # Walk forward through the holding period
        barrier_hit = "time"
        barrier_ret = 0.0
        exit_date = close.index[min(i + cfg.max_holding_days, len(close) - 1)]

        for j in range(1, cfg.max_holding_days + 1):
            if i + j >= len(close):
                break

            current_ret = (close.iloc[i + j] / entry_price) - 1
            exit_date = close.index[i + j]

            # Check profit barrier (upper)
            if current_ret >= pt_threshold:
                barrier_hit = "profit"
                barrier_ret = current_ret
                break

            # Check stop-loss barrier (lower)
            if current_ret <= -sl_threshold:
                barrier_hit = "stop"
                barrier_ret = current_ret
                break

            barrier_ret = current_ret

        # Assign label
        if barrier_hit == "profit":
            label = 1
        elif barrier_hit == "stop":
            label = -1
        else:
            label = 0

        labels.append({
            "date": entry_date,
            "label": label,
            "ret": barrier_ret,
            "barrier": barrier_hit,
            "t_end": exit_date,
        })

    result = pd.DataFrame(labels).set_index("date")

    # Log distribution
    dist = result["label"].value_counts().to_dict()
    print(f"  Labels: +1={dist.get(1,0)}, 0={dist.get(0,0)}, -1={dist.get(-1,0)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE WEIGHTS — Average Uniqueness
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sample_weights(label_df: pd.DataFrame, close: pd.Series) -> pd.Series:
    """
    Compute sample weights based on average uniqueness.
    From: Advances in Financial Machine Learning (de Prado, Ch. 4).

    Overlapping labels (e.g., label at day 1 spans to day 10, label at day 5
    spans to day 15) share information. Weight each sample inversely to
    its concurrency with other samples.
    """
    dates = close.index
    t_start = label_df.index
    t_end = label_df["t_end"]

    # Build concurrency matrix: for each time step, count how many labels
    # are "active" (between their start and end)
    n = len(label_df)
    uniqueness = np.ones(n)

    for i in range(n):
        start_i = t_start[i]
        end_i = t_end.iloc[i]

        # Count concurrent labels
        concurrent = 0
        for j in range(n):
            start_j = t_start[j]
            end_j = t_end.iloc[j]
            # Check overlap
            if start_j <= end_i and end_j >= start_i:
                concurrent += 1

        uniqueness[i] = 1.0 / max(concurrent, 1)

    weights = pd.Series(uniqueness, index=label_df.index, name="sample_weight")

    # Normalize weights to sum to len(labels)
    weights = weights * len(weights) / weights.sum()

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  META-LABELING
# ═══════════════════════════════════════════════════════════════════════════════

def meta_labels(primary_labels: pd.Series) -> pd.Series:
    """
    Convert triple-barrier labels to meta-labels for a long-only strategy.

    In a long-only regime:
      - Primary model says "go long" on every bar
      - Meta-label = 1 if the long trade was profitable (label == 1)
      - Meta-label = 0 if it was unprofitable (label == -1 or 0)

    The meta-model then learns WHEN to trust the primary signal,
    and its probability output drives position sizing.
    """
    return (primary_labels == 1).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_labels(
    close: pd.Series,
    cfg: LabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Full labeling pipeline.

    Returns:
      - label_df: DataFrame with label, ret, barrier, t_end columns
      - sample_weights: Series of sample weights
      - meta: Series of binary meta-labels
    """
    cfg = cfg or LabelConfig()

    print(f"  ── Labeling ──")
    print(f"  Barriers: PT={cfg.profit_taking}, SL={cfg.stop_loss}, max_days={cfg.max_holding_days}")
    print(f"  Dynamic barriers: {cfg.use_dynamic_barriers}")

    # 1. Triple barrier labels
    label_df = triple_barrier_labels(close, cfg)

    # 2. Sample weights
    if cfg.compute_sample_weights:
        print(f"  Computing sample weights...")
        weights = compute_sample_weights(label_df, close)
    else:
        weights = pd.Series(1.0, index=label_df.index, name="sample_weight")

    # 3. Meta-labels
    meta = meta_labels(label_df["label"])
    meta_dist = meta.value_counts().to_dict()
    print(f"  Meta-labels: profitable={meta_dist.get(1,0)}, unprofitable={meta_dist.get(0,0)}")

    return label_df, weights, meta


if __name__ == "__main__":
    from data_pipeline import load_parquet
    raw = load_parquet("raw_merged")
    label_df, weights, meta = build_labels(raw["close"])
    print(f"\nLabel distribution:\n{label_df['label'].value_counts()}")
    print(f"\nSample weights stats:\n{weights.describe()}")
