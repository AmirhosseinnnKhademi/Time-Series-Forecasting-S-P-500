from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, drop duplicates, and forward-fill gaps."""
    df = df.copy().sort_index().drop_duplicates()
    df = df.ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Add MAs, EMA, RSI, MACD, Bollinger Bands, returns, and volatility."""
    df = df.copy()
    close = df[close_col]

    # Simple moving averages
    for w in [7, 21, 50, 200]:
        df[f"MA_{w}"] = close.rolling(w).mean()

    # EMAs
    for span in [12, 26]:
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    # RSI-14
    df["RSI_14"] = _rsi(close, 14)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands (20-day, 2 std)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Returns
    for d in [1, 5, 20]:
        df[f"Return_{d}d"] = close.pct_change(d)

    # Annualised historical volatility
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std() * np.sqrt(252)

    # Volume features
    if "Volume" in df.columns:
        df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA_20"]

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_lag_features(
    df: pd.DataFrame,
    col: str = "Close",
    lags: List[int] | None = None,
) -> pd.DataFrame:
    """Append lagged values of *col* as new columns."""
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Train / val / test split  (chronological — no shuffle)
# ---------------------------------------------------------------------------

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split preserving temporal order."""
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler on train only, then transform all splits."""
    scaler = MinMaxScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        scaler,
    )


# ---------------------------------------------------------------------------
# Sequence creation for LSTM
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    seq_len: int,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of *seq_len* over *data* to create (X, y) pairs.

    Args:
        data:       2-D array of shape (timesteps, features).
        seq_len:    Look-back window length.
        target_idx: Column index of the variable to predict.

    Returns:
        X of shape (n_samples, seq_len, n_features).
        y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# High-level pipeline helpers
# ---------------------------------------------------------------------------

def prepare_ml_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Close",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """Prepare data for tree-based / linear models (flat feature matrix).

    Returns a dict with keys:
        X_train, X_val, X_test  (scaled)
        y_train, y_val, y_test  (scaled)
        y_train_raw, y_val_raw, y_test_raw  (original scale)
        feature_scaler, target_scaler
        train_df, val_df, test_df
    """
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    train_df, val_df, test_df = train_val_test_split(df, train_ratio, val_ratio)

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    y_train_raw = train_df[target_col].values
    y_val_raw   = val_df[target_col].values
    y_test_raw  = test_df[target_col].values

    X_train_s, X_val_s, X_test_s, feat_scaler = scale_features(X_train, X_val, X_test)

    target_scaler = MinMaxScaler()
    y_train_s = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val_s   = target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test_s  = target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    return dict(
        X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
        y_train=y_train_s, y_val=y_val_s, y_test=y_test_s,
        y_train_raw=y_train_raw, y_val_raw=y_val_raw, y_test_raw=y_test_raw,
        feature_scaler=feat_scaler, target_scaler=target_scaler,
        train_df=train_df, val_df=val_df, test_df=test_df,
    )


def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Close",
    seq_len: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """Prepare sequences for an LSTM model.

    Returns a dict with keys:
        X_train, X_val, X_test  (3-D arrays: samples × seq_len × features)
        y_train, y_val, y_test  (scaled targets)
        scaler, all_cols, target_idx
        train_df, val_df, test_df
    """
    # Ensure target_col is in all_cols at position 0
    all_cols = [target_col] + [c for c in feature_cols if c != target_col]
    df = df.dropna(subset=all_cols).copy()
    train_df, val_df, test_df = train_val_test_split(df, train_ratio, val_ratio)

    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train_df[all_cols])
    val_s   = scaler.transform(val_df[all_cols])
    test_s  = scaler.transform(test_df[all_cols])

    target_idx = 0  # Close is always index 0
    X_train, y_train = create_sequences(train_s, seq_len, target_idx)
    X_val,   y_val   = create_sequences(val_s,   seq_len, target_idx)
    # Prefix test sequences with the last seq_len rows of val so that the
    # first test prediction starts at test_df.index[0] — matching the
    # ARIMA / XGBoost test window exactly (no dates consumed as warm-up).
    combined_for_test = np.concatenate([val_s[-seq_len:], test_s], axis=0)
    X_test,  y_test  = create_sequences(combined_for_test, seq_len, target_idx)

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        scaler=scaler, all_cols=all_cols, target_idx=target_idx,
        seq_len=seq_len,
        train_df=train_df, val_df=val_df, test_df=test_df,
    )
