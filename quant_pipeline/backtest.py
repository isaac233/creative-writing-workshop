"""
Backtest — Walk-forward simulation with transaction costs.

Features:
  - Event-driven: processes signals day-by-day
  - Transaction costs: commission + slippage
  - Position sizing via meta-model confidence
  - Equity curve, drawdown, and performance metrics
  - Long-only constraint enforced
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import BacktestConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    total_costs: float = 0.0
    exposure_pct: float = 0.0

    def summary(self) -> str:
        return (
            f"\n  ── Backtest Results ──\n"
            f"  Total Return:    {self.total_return:>10.2%}\n"
            f"  Annual Return:   {self.annual_return:>10.2%}\n"
            f"  Sharpe Ratio:    {self.sharpe_ratio:>10.2f}\n"
            f"  Sortino Ratio:   {self.sortino_ratio:>10.2f}\n"
            f"  Max Drawdown:    {self.max_drawdown:>10.2%}\n"
            f"  Max DD Duration: {self.max_drawdown_duration:>10d} days\n"
            f"  Win Rate:        {self.win_rate:>10.2%}\n"
            f"  Profit Factor:   {self.profit_factor:>10.2f}\n"
            f"  Total Trades:    {self.total_trades:>10d}\n"
            f"  Avg Trade Ret:   {self.avg_trade_return:>10.4%}\n"
            f"  Total Costs:     ${self.total_costs:>10,.2f}\n"
            f"  Time in Market:  {self.exposure_pct:>10.1%}\n"
        )


def compute_metrics(
    equity: pd.Series,
    trades: list[dict],
    initial_capital: float,
    risk_free: float = 0.05,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from equity curve."""
    m = PerformanceMetrics()

    if equity.empty or len(equity) < 2:
        return m

    # Returns
    daily_ret = equity.pct_change().dropna()
    n_days = len(daily_ret)
    n_years = n_days / 252

    m.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    m.annual_return = (1 + m.total_return) ** (1 / max(n_years, 0.01)) - 1

    # Risk
    excess_ret = daily_ret - risk_free / 252
    if daily_ret.std() > 0:
        m.sharpe_ratio = excess_ret.mean() / daily_ret.std() * np.sqrt(252)

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        m.sortino_ratio = excess_ret.mean() / downside.std() * np.sqrt(252)

    # Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    m.max_drawdown = drawdown.min()

    # Drawdown duration
    underwater = drawdown < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_lengths = underwater.groupby(groups).sum()
        m.max_drawdown_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Trade stats
    if trades:
        m.total_trades = len(trades)
        returns = [t["return"] for t in trades]
        m.avg_trade_return = np.mean(returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        m.win_rate = len(wins) / max(len(returns), 1)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        m.profit_factor = gross_profit / gross_loss

        m.total_costs = sum(t.get("cost", 0) for t in trades)

    # Exposure
    invested_days = sum(1 for t in trades for _ in range(t.get("holding_days", 1)))
    m.exposure_pct = min(invested_days / max(n_days, 1), 1.0)

    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    close: pd.Series,
    signals: pd.Series,
    confidence: pd.Series,
    cfg: BacktestConfig | None = None,
) -> dict:
    """
    Event-driven backtest.

    Args:
      close: daily close prices
      signals: 1 = enter long, 0 = stay out, -1 = exit
      confidence: P(profitable) from meta-model, used for position sizing
      cfg: backtest configuration

    Returns dict with equity curve, trades, metrics.
    """
    cfg = cfg or BacktestConfig()

    # Align
    common = close.index.intersection(signals.index).intersection(confidence.index)
    close = close.loc[common]
    signals = signals.loc[common]
    confidence = confidence.loc[common]

    capital = cfg.initial_capital
    position = 0.0       # shares held
    entry_price = 0.0
    entry_date = None

    equity_curve = []
    trades = []

    for i in range(len(close)):
        date = close.index[i]
        price = close.iloc[i]
        signal = signals.iloc[i]
        conf = confidence.iloc[i]

        # Current portfolio value
        portfolio_value = capital + position * price

        # ── Exit logic ────────────────────────────────────────────────────
        if position > 0 and signal <= 0:
            # Sell
            proceeds = position * price
            cost = proceeds * (cfg.commission_pct + cfg.slippage_pct)
            capital += proceeds - cost

            trade_ret = (price / entry_price) - 1 - cfg.commission_pct * 2 - cfg.slippage_pct * 2
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": price,
                "return": trade_ret,
                "cost": cost,
                "holding_days": (date - entry_date).days if entry_date else 0,
                "confidence": conf,
            })

            position = 0.0
            entry_price = 0.0
            entry_date = None
            portfolio_value = capital

        # ── Entry logic ───────────────────────────────────────────────────
        elif position == 0 and signal == 1 and conf >= cfg.min_confidence:
            # Size position by confidence
            allocation = min(conf, cfg.max_position_pct)
            invest_amount = capital * allocation

            cost = invest_amount * (cfg.commission_pct + cfg.slippage_pct)
            shares = (invest_amount - cost) / price

            position = shares
            entry_price = price
            entry_date = date
            capital -= invest_amount

            portfolio_value = capital + position * price

        equity_curve.append({"date": date, "equity": portfolio_value})

    # Close any remaining position at end
    if position > 0:
        final_price = close.iloc[-1]
        proceeds = position * final_price
        cost = proceeds * (cfg.commission_pct + cfg.slippage_pct)
        capital += proceeds - cost
        trade_ret = (final_price / entry_price) - 1
        trades.append({
            "entry_date": entry_date,
            "exit_date": close.index[-1],
            "entry_price": entry_price,
            "exit_price": final_price,
            "return": trade_ret,
            "cost": cost,
            "holding_days": (close.index[-1] - entry_date).days,
            "confidence": 0,
        })

    equity = pd.DataFrame(equity_curve).set_index("date")["equity"]
    metrics = compute_metrics(equity, trades, cfg.initial_capital, cfg.risk_free_rate)

    return {
        "equity": equity,
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "metrics": metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION FROM MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    stage1_model,
    stage2_model,
    X: pd.DataFrame,
    label_map_inv: dict,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate trading signals from the two-stage model.

    Stage 1 predicts direction (triple barrier).
    Stage 2 predicts confidence (meta-label).

    Long-only: signal=1 only when stage1 predicts profit (class 2).
    Confidence = stage2 probability.
    """
    # Stage 1: direction
    stage1_pred = stage1_model.predict(X)
    direction = pd.Series(
        [label_map_inv.get(p, 0) for p in stage1_pred],
        index=X.index,
    )

    # Signal: 1 when direction is profit, 0 otherwise (long-only)
    signals = (direction == 1).astype(int)

    # Stage 2: confidence
    meta_prob = stage2_model.predict_proba(X)[:, 1]
    confidence = pd.Series(meta_prob, index=X.index)

    return signals, confidence


# ═══════════════════════════════════════════════════════════════════════════════
#  BUY-AND-HOLD BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_buy_hold(
    close: pd.Series,
    initial_capital: float = 100_000,
) -> dict:
    """Simple buy-and-hold benchmark for comparison."""
    shares = initial_capital / close.iloc[0]
    equity = shares * close
    equity.name = "equity"

    metrics = compute_metrics(equity, [], initial_capital)
    return {"equity": equity, "metrics": metrics}
