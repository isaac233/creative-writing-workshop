"""
Data Pipeline — Fetch, clean, and store market + alternative data.

Sources:
  - yfinance: SPY OHLCV, sector ETFs, VIX
  - FRED API: economic indicators (yield curve, credit spreads, etc.)

Storage: DuckDB + Parquet for analytical queries.
"""
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

from config import DataConfig, DATA_DIR, PARQUET_DIR, DUCKDB_PATH

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    logger.info(f"Fetching {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Clean
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()

    logger.info(f"  {ticker}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_sector_etfs(etfs: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices for sector/factor ETFs."""
    logger.info(f"Fetching {len(etfs)} ETFs...")
    frames = {}
    for etf in etfs:
        try:
            df = yf.download(etf, start=start, end=end, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                frames[etf] = df["Close"]
        except Exception as e:
            logger.warning(f"  Skipping {etf}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    result = result.ffill().dropna()
    logger.info(f"  ETFs: {len(result)} rows, {len(result.columns)} tickers")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FRED ECONOMIC DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fred_data(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch economic data from FRED via pandas-datareader.
    Falls back to yfinance Treasury data if datareader fails.
    """
    frames = {}

    # Try pandas-datareader first
    try:
        import pandas_datareader.data as web
        for sid in series_ids:
            try:
                s = web.DataReader(sid, "fred", start, end)
                if not s.empty:
                    frames[sid] = s.iloc[:, 0]
                    logger.info(f"  FRED {sid}: {len(s)} rows")
            except Exception as e:
                logger.warning(f"  FRED {sid} failed: {e}")
    except ImportError:
        logger.warning("  pandas-datareader not available; fetching VIX from yfinance")
        # Minimal fallback: at least get VIX
        try:
            vix = yf.download("^VIX", start=start, end=end, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if not vix.empty:
                frames["VIXCLS"] = vix["Close"]
        except Exception:
            pass

    if not frames:
        logger.warning("  No FRED data retrieved")
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    # Forward-fill economic data (released at different frequencies)
    result = result.ffill()
    logger.info(f"  FRED total: {len(result)} rows, {len(result.columns)} series")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE & ALIGN
# ═══════════════════════════════════════════════════════════════════════════════

def merge_datasets(
    ohlcv: pd.DataFrame,
    etfs: pd.DataFrame,
    fred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all data sources on the trading calendar (SPY dates).
    Forward-fills alternative data to align with market days.
    """
    base = ohlcv.copy()

    if not etfs.empty:
        etf_cols = {c: f"etf_{c}" for c in etfs.columns}
        etfs = etfs.rename(columns=etf_cols)
        base = base.join(etfs, how="left")

    if not fred.empty:
        fred_cols = {c: f"fred_{c}" for c in fred.columns}
        fred = fred.rename(columns=fred_cols)
        base = base.join(fred, how="left")

    # Forward-fill alt data, then drop rows with NaN in core OHLCV
    base = base.ffill()
    base = base.dropna(subset=["open", "high", "low", "close", "volume"])

    logger.info(f"Merged dataset: {len(base)} rows × {len(base.columns)} columns")
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  STORAGE — DuckDB + Parquet
# ═══════════════════════════════════════════════════════════════════════════════

def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame to Parquet."""
    path = PARQUET_DIR / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow")
    logger.info(f"Saved {path.name}: {len(df)} rows × {len(df.columns)} cols")
    return path


def load_parquet(name: str) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    path = PARQUET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def init_duckdb(df: pd.DataFrame, table_name: str = "raw_data") -> None:
    """Write merged data into DuckDB for analytical queries."""
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    con.close()
    logger.info(f"DuckDB '{table_name}': {count} rows written to {DUCKDB_PATH.name}")


def query_duckdb(sql: str) -> pd.DataFrame:
    """Run an analytical query against DuckDB."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    result = con.execute(sql).fetchdf()
    con.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: DataConfig | None = None) -> pd.DataFrame:
    """
    Full data pipeline: fetch → merge → store.
    Returns the merged DataFrame.
    """
    cfg = cfg or DataConfig()

    print(f"\n  ── Data Pipeline ──")
    print(f"  Ticker: {cfg.ticker}")
    print(f"  Range:  {cfg.start_date} → {cfg.end_date}")

    # 1. Fetch core OHLCV
    ohlcv = fetch_ohlcv(cfg.ticker, cfg.start_date, cfg.end_date, cfg.interval)

    # 2. Fetch sector ETFs
    etfs = fetch_sector_etfs(cfg.sector_etfs, cfg.start_date, cfg.end_date)

    # 3. Fetch FRED economic data
    fred = fetch_fred_data(cfg.fred_series, cfg.start_date, cfg.end_date)

    # 4. Merge
    merged = merge_datasets(ohlcv, etfs, fred)

    # 5. Store
    save_parquet(merged, "raw_merged")
    init_duckdb(merged, "raw_data")

    print(f"  ✓ Pipeline complete: {len(merged)} rows × {len(merged.columns)} columns\n")
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = run_pipeline()
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
