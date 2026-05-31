#!/usr/bin/env python3
"""
Main — End-to-end quantitative pipeline orchestrator.

Stages:
  1. Data ingestion (market + economic + sector data)
  2. Feature engineering (stationary transforms)
  3. Regime detection
  4. Triple barrier labeling + sample weights
  5. Stage 1 model: triple barrier classifier
  6. SHAP feature pruning
  7. Stage 2 model: meta-label confidence
  8. Walk-forward backtest with transaction costs
  9. Performance comparison vs baselines
"""
import logging
import sys
import time

import pandas as pd
import numpy as np

from config import (
    DataConfig, FeatureConfig, LabelConfig, ModelConfig, BacktestConfig,
    PARQUET_DIR,
)
from data_pipeline import run_pipeline, save_parquet, load_parquet
from features import build_features
from labels import build_labels
from regime import build_regime_features
from modeling import run_modeling, run_baselines, generate_signals
from backtest import run_backtest, benchmark_buy_hold

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)


def main():
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     Quantitative Pipeline — SPY Long-Only Strategy           ║
║     XGBoost + Triple Barrier + Meta-Labeling                 ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Configuration ─────────────────────────────────────────────────────────
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    label_cfg = LabelConfig()
    model_cfg = ModelConfig()
    bt_cfg = BacktestConfig()

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 1: DATA INGESTION
    # ══════════════════════════════════════════════════════════════════════════
    print("━" * 60)
    print("  STAGE 1: Data Ingestion")
    print("━" * 60)

    try:
        raw = load_parquet("raw_merged")
        print(f"  ✓ Loaded cached data: {len(raw)} rows × {len(raw.columns)} cols")
    except FileNotFoundError:
        raw = run_pipeline(data_cfg)

    close = raw["close"]

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 2: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    print("━" * 60)
    print("  STAGE 2: Feature Engineering")
    print("━" * 60)

    features = build_features(raw, feat_cfg)

    # Add regime features
    regimes = build_regime_features(close)
    # Align and merge
    common = features.index.intersection(regimes.index)
    features = features.loc[common]
    regime_features = regimes.loc[common]
    features = pd.concat([features, regime_features], axis=1)

    # Drop any remaining NaN columns
    features = features.dropna(axis=1, how="all").fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    save_parquet(features, "features")
    print(f"  ✓ Final feature matrix: {features.shape[0]} rows × {features.shape[1]} columns")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 3: LABELING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  STAGE 3: Triple Barrier Labeling")
    print("━" * 60)

    label_df, sample_weights, meta_y = build_labels(close, label_cfg)
    save_parquet(label_df, "labels")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 4: MODELING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  STAGE 4: Modeling")
    print("━" * 60)

    results = run_modeling(
        X=features,
        labels=label_df,
        sample_weights=sample_weights,
        meta_y=meta_y,
        close=close,
        cfg=model_cfg,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 5: BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  STAGE 5: Walk-Forward Backtest")
    print("━" * 60)

    # Generate signals on the pruned feature set
    X_pruned = results["X_pruned"]
    stage1_model = results["stage1"]["model"]
    stage2_model = results["stage2"]["model"]
    label_map_inv = results["stage1"]["label_map_inv"]

    signals, confidence = generate_signals(
        stage1_model, stage2_model, X_pruned, label_map_inv
    )

    # Align with close prices
    common_bt = close.index.intersection(signals.index)
    bt_result = run_backtest(
        close.loc[common_bt],
        signals.loc[common_bt],
        confidence.loc[common_bt],
        bt_cfg,
    )

    print(bt_result["metrics"].summary())

    # Benchmark
    bh = benchmark_buy_hold(close.loc[common_bt], bt_cfg.initial_capital)
    print(f"  ── Buy & Hold Benchmark ──")
    print(f"  Total Return:  {bh['metrics'].total_return:.2%}")
    print(f"  Sharpe Ratio:  {bh['metrics'].sharpe_ratio:.2f}")
    print(f"  Max Drawdown:  {bh['metrics'].max_drawdown:.2%}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 6: SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("  STAGE 6: Saving Results")
    print("━" * 60)

    save_parquet(bt_result["equity"].to_frame(), "equity_curve")
    if not bt_result["trades"].empty:
        save_parquet(bt_result["trades"], "trades")

    # Feature importance
    importance = results["stage1"]["feature_importance"]
    importance.to_frame("importance").to_parquet(PARQUET_DIR / "feature_importance.parquet")

    # CV results
    results["stage1"]["cv_results"].to_parquet(PARQUET_DIR / "cv_results_stage1.parquet")
    results["stage2"]["cv_results"].to_parquet(PARQUET_DIR / "cv_results_stage2.parquet")

    elapsed = time.time() - t0
    print(f"\n  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {PARQUET_DIR}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Data:       {len(raw)} days, {len(raw.columns)} raw columns")
    print(f"  Features:   {features.shape[1]} total → {len(results['top_features'])} after pruning")
    print(f"  Labels:     {len(label_df)} samples")
    s1 = results["stage1"]["cv_results"]
    s2 = results["stage2"]["cv_results"]
    print(f"  Stage 1 CV: F1={s1['f1_weighted'].mean():.3f} ± {s1['f1_weighted'].std():.3f}")
    print(f"  Stage 2 CV: F1={s2['f1'].mean():.3f} ± {s2['f1'].std():.3f}")
    print(f"  Backtest:   Sharpe={bt_result['metrics'].sharpe_ratio:.2f}, "
          f"Return={bt_result['metrics'].total_return:.2%}")
    print(f"  vs B&H:     Sharpe={bh['metrics'].sharpe_ratio:.2f}, "
          f"Return={bh['metrics'].total_return:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
