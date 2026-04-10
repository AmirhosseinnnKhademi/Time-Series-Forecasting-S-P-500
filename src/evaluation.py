from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE ignoring zero-valued actuals."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (avoids division-by-zero issues)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Return MAE, RMSE, MAPE, SMAPE, R² and directional accuracy.

    Args:
        y_true:     Ground-truth values (original scale).
        y_pred:     Model predictions  (original scale).
        model_name: Label used in the returned dict.

    Returns:
        dict with metric names as keys.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape  = _safe_mape(y_true, y_pred)
    smape = _smape(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    da = directional_accuracy(y_true, y_pred)

    return {
        "Model":  model_name,
        "MAE":    round(mae,   4),
        "RMSE":   round(rmse,  4),
        "MAPE":   round(mape,  4),
        "SMAPE":  round(smape, 4),
        "R²":     round(r2,    4),
        "Dir.Acc": round(da,   4),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of timesteps where the predicted direction matches the actual.

    Direction is defined as sign(y[t] - y[t-1]).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(y_true))
    pred_dir   = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir))


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(results: list[dict]) -> pd.DataFrame:
    """Combine a list of metric dicts into a sorted comparison table.

    Args:
        results: List of dicts returned by :func:`compute_metrics`.

    Returns:
        DataFrame sorted by RMSE (ascending).
    """
    df = pd.DataFrame(results).set_index("Model")
    return df.sort_values("RMSE")


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validation(
    series: pd.Series,
    initial_train_size: int,
    order: tuple = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling one-step-ahead ARIMA evaluation.

    Returns:
        y_true: actual test observations.
        y_pred: one-step-ahead forecasts.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    values = series.values
    history = list(values[:initial_train_size])
    predictions = []

    for t in range(initial_train_size, len(values)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(history, order=order).fit()
        yhat = fitted.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(values[t])

    return values[initial_train_size:], np.array(predictions)


# ---------------------------------------------------------------------------
# Inverse-scale helper
# ---------------------------------------------------------------------------

def inverse_scale(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """Invert MinMaxScaler transformation for a 1-D prediction array."""
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
