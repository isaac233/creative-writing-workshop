"""
Modeling — XGBoost with Purged Walk-Forward CV, SHAP Pruning, Meta-Labeling.

Architecture:
  Stage 1: Triple-barrier classifier (multi:softprob, 3 classes)
  Stage 2: Meta-label classifier (binary:logistic) — predicts P(profitable)
  Feature pruning: SHAP importance → keep top N features
  CV: Purged + embargoed walk-forward splits (no standard KFold)
  Baselines: buy-and-hold, SMA crossover, logistic regression
"""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss, f1_score
)
import xgboost as xgb

from config import ModelConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
#  PURGED WALK-FORWARD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class PurgedWalkForwardCV:
    """
    Walk-forward time-series cross-validation with purging and embargo.

    - Train window: fixed-size sliding window
    - Purge gap: removes samples between train and test to prevent leakage
      from overlapping triple-barrier labels
    - Embargo: removes a fraction of the end of each training set to prevent
      serial correlation leakage

    From: Advances in Financial Machine Learning (de Prado, Ch. 7).
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_window: int = 504,
        test_window: int = 63,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame):
        """
        Yield (train_indices, test_indices) for each fold.
        """
        n = len(X)
        total_needed = self.train_window + self.purge_gap + self.test_window
        step = max(1, (n - total_needed) // self.n_splits)

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - 1 - i) * step
            test_start = test_end - self.test_window

            if test_start < 0:
                continue

            train_end = test_start - self.purge_gap
            train_start = max(0, train_end - self.train_window)

            if train_start >= train_end or train_end <= 0:
                continue

            # Apply embargo to the end of the training set
            embargo_size = int(self.embargo_pct * (train_end - train_start))
            train_end_actual = train_end - embargo_size

            if train_start >= train_end_actual:
                continue

            train_idx = np.arange(train_start, train_end_actual)
            test_idx = np.arange(test_start, min(test_end, n))

            yield train_idx, test_idx

    def get_n_splits(self):
        return self.n_splits


# ═══════════════════════════════════════════════════════════════════════════════
#  XGBOOST TRAINING — Stage 1 (Triple Barrier)
# ═══════════════════════════════════════════════════════════════════════════════

def train_triple_barrier_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series,
    cfg: ModelConfig | None = None,
) -> dict:
    """
    Train XGBoost multi-class model on triple-barrier labels.
    Uses purged walk-forward CV. Returns results dict.

    Labels: -1 (stop), 0 (time), 1 (profit) → mapped to 0, 1, 2 for XGBoost.
    """
    cfg = cfg or ModelConfig()

    # Map labels: {-1: 0, 0: 1, 1: 2} for multi:softprob
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    cv = PurgedWalkForwardCV(
        n_splits=cfg.n_splits,
        train_window=cfg.train_window,
        test_window=cfg.test_window,
        purge_gap=cfg.purge_gap,
        embargo_pct=cfg.embargo_pct,
    )

    fold_results = []
    all_importances = []
    best_model = None
    best_score = -np.inf

    print(f"  ── Stage 1: Triple Barrier Model ──")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  CV: {cfg.n_splits} purged walk-forward splits")

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_mapped.iloc[train_idx]
        y_test = y_mapped.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]

        model = xgb.XGBClassifier(**cfg.xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train.values,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        try:
            ll = log_loss(y_test, y_prob, labels=[0, 1, 2])
        except Exception:
            ll = np.nan

        fold_results.append({
            "fold": fold_i,
            "accuracy": acc,
            "f1_weighted": f1,
            "log_loss": ll,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

        # Feature importance
        importance = pd.Series(
            model.feature_importances_, index=X.columns, name=f"fold_{fold_i}"
        )
        all_importances.append(importance)

        if f1 > best_score:
            best_score = f1
            best_model = model

        print(f"    Fold {fold_i}: acc={acc:.3f}, F1={f1:.3f}, log_loss={ll:.4f}")

    # Aggregate importance across folds
    importance_df = pd.concat(all_importances, axis=1)
    mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)

    results_df = pd.DataFrame(fold_results)
    print(f"  Mean: acc={results_df['accuracy'].mean():.3f}, "
          f"F1={results_df['f1_weighted'].mean():.3f}")

    return {
        "model": best_model,
        "cv_results": results_df,
        "feature_importance": mean_importance,
        "label_map": {-1: 0, 0: 1, 1: 2},
        "label_map_inv": {0: -1, 1: 0, 2: 1},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  XGBOOST TRAINING — Stage 2 (Meta-Label)
# ═══════════════════════════════════════════════════════════════════════════════

def train_meta_model(
    X: pd.DataFrame,
    meta_y: pd.Series,
    sample_weights: pd.Series,
    cfg: ModelConfig | None = None,
) -> dict:
    """
    Train binary XGBoost meta-model: P(primary signal is profitable).
    Output probability drives position sizing in the backtest.
    """
    cfg = cfg or ModelConfig()

    meta_params = cfg.xgb_params.copy()
    meta_params["objective"] = "binary:logistic"
    meta_params.pop("num_class", None)

    cv = PurgedWalkForwardCV(
        n_splits=cfg.n_splits,
        train_window=cfg.train_window,
        test_window=cfg.test_window,
        purge_gap=cfg.purge_gap,
        embargo_pct=cfg.embargo_pct,
    )

    fold_results = []
    best_model = None
    best_score = -np.inf

    print(f"\n  ── Stage 2: Meta-Label Model ──")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = meta_y.iloc[train_idx]
        y_test = meta_y.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]

        model = xgb.XGBClassifier(**meta_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train.values,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        fold_results.append({
            "fold": fold_i,
            "accuracy": acc,
            "f1": f1,
            "mean_prob": y_prob.mean(),
        })

        if f1 > best_score:
            best_score = f1
            best_model = model

        print(f"    Fold {fold_i}: acc={acc:.3f}, F1={f1:.3f}, mean_P={y_prob.mean():.3f}")

    results_df = pd.DataFrame(fold_results)
    print(f"  Mean: acc={results_df['accuracy'].mean():.3f}, "
          f"F1={results_df['f1'].mean():.3f}")

    return {
        "model": best_model,
        "cv_results": results_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP FEATURE PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

def prune_features_shap(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    max_features: int = 100,
    sample_size: int = 500,
) -> list[str]:
    """
    Use SHAP values to identify and keep the top-N most impactful features.
    Falls back to XGBoost built-in importance if SHAP fails.
    """
    print(f"\n  ── Feature Pruning (target: {max_features}) ──")

    try:
        import shap
        # Sample for speed
        X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For multi-class, shap_values is a list of arrays
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        importance = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
        method = "SHAP"

    except Exception as e:
        logger.warning(f"  SHAP failed ({e}), using built-in importance")
        importance = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        method = "XGBoost gain"

    top_features = importance.head(max_features).index.tolist()
    pruned = len(X.columns) - len(top_features)
    print(f"  {method}: keeping {len(top_features)} features, pruned {pruned}")
    print(f"  Top 10: {', '.join(top_features[:10])}")

    return top_features


# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold SPY returns as baseline."""
    total_ret = (close.iloc[-1] / close.iloc[0]) - 1
    n_years = len(close) / 252
    annual_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    log_ret = np.log(close / close.shift(1)).dropna()
    sharpe = log_ret.mean() / log_ret.std() * np.sqrt(252)

    return {
        "strategy": "Buy & Hold",
        "total_return": total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
    }


def baseline_sma_crossover(close: pd.Series, short: int = 50, long: int = 200) -> dict:
    """SMA crossover: long when SMA_short > SMA_long, else cash."""
    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()
    signal = (sma_short > sma_long).astype(float)

    log_ret = np.log(close / close.shift(1))
    strat_ret = signal.shift(1) * log_ret  # lag signal
    strat_ret = strat_ret.dropna()

    total_ret = np.exp(strat_ret.sum()) - 1
    n_years = len(strat_ret) / 252
    annual_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0

    return {
        "strategy": f"SMA {short}/{long}",
        "total_return": total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
    }


def baseline_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ModelConfig | None = None,
) -> dict:
    """Logistic regression baseline with same CV."""
    cfg = cfg or ModelConfig()
    cv = PurgedWalkForwardCV(
        n_splits=cfg.n_splits,
        train_window=cfg.train_window,
        test_window=cfg.test_window,
        purge_gap=cfg.purge_gap,
    )

    accs = []
    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx].fillna(0)
        X_test = X.iloc[test_idx].fillna(0)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, lr.predict(X_test)))

    return {
        "strategy": "Logistic Regression",
        "mean_accuracy": np.mean(accs),
        "std_accuracy": np.std(accs),
    }


def run_baselines(close: pd.Series, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run all baselines and return comparison table."""
    print(f"\n  ── Baselines ──")

    bh = baseline_buy_and_hold(close)
    sma = baseline_sma_crossover(close)
    lr = baseline_logistic(X, y)

    print(f"  Buy & Hold:  Sharpe={bh['sharpe']:.2f}, Annual={bh['annual_return']:.1%}")
    print(f"  SMA 50/200:  Sharpe={sma['sharpe']:.2f}, Annual={sma['annual_return']:.1%}")
    print(f"  Logistic:    Acc={lr['mean_accuracy']:.3f} ± {lr['std_accuracy']:.3f}")

    return pd.DataFrame([bh, sma, lr])


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL MODELING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_modeling(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    sample_weights: pd.Series,
    meta_y: pd.Series,
    close: pd.Series,
    cfg: ModelConfig | None = None,
) -> dict:
    """
    Full two-stage modeling pipeline:
      1. Train triple-barrier model on all features
      2. Prune features via SHAP
      3. Retrain on pruned features
      4. Train meta-label model
      5. Compare against baselines
    """
    cfg = cfg or ModelConfig()

    # Align all inputs
    common_idx = X.index.intersection(labels.index).intersection(sample_weights.index)
    X = X.loc[common_idx]
    y = labels.loc[common_idx, "label"]
    w = sample_weights.loc[common_idx]
    meta = meta_y.loc[common_idx]
    close_aligned = close.loc[close.index.isin(common_idx)]

    print(f"\n{'='*60}")
    print(f"  MODELING PIPELINE")
    print(f"  Aligned samples: {len(common_idx)}")
    print(f"  Features: {X.shape[1]}")
    print(f"{'='*60}")

    # Stage 1: Initial training
    stage1 = train_triple_barrier_model(X, y, w, cfg)

    # Feature pruning
    top_features = prune_features_shap(
        stage1["model"], X,
        max_features=cfg.max_features,
        sample_size=cfg.shap_sample_size,
    )
    X_pruned = X[top_features]

    # Retrain on pruned features
    print(f"\n  ── Retrain on {len(top_features)} features ──")
    stage1_final = train_triple_barrier_model(X_pruned, y, w, cfg)

    # Stage 2: Meta-label model
    stage2 = train_meta_model(X_pruned, meta, w, cfg)

    # Baselines
    baselines = run_baselines(close_aligned, X_pruned, y.map({-1: 0, 0: 1, 1: 2}))

    return {
        "stage1": stage1_final,
        "stage2": stage2,
        "top_features": top_features,
        "baselines": baselines,
        "X_pruned": X_pruned,
    }
