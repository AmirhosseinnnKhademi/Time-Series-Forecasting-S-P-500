"""DVC Stage 4 — Generate and save evaluation plots.

Reads  : models/*, data/processed/*
Writes : reports/plots/*.png

Run directly:   python src/pipeline/evaluate.py
Run via DVC:    dvc repro evaluate
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import compute_metrics, inverse_scale  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_forecast,
    plot_model_comparison,
    plot_residuals,
)


def _invert_lstm(pred_scaled, scaler, target_idx, n_features):
    dummy = np.zeros((len(pred_scaled), n_features))
    dummy[:, target_idx] = pred_scaled
    return scaler.inverse_transform(dummy)[:, target_idx]


def main() -> None:
    params   = yaml.safe_load((ROOT / "params.yaml").read_text())
    proc_dir = ROOT / "data" / "processed"
    mdl_dir  = ROOT / "models"
    plots_dir = ROOT / "reports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    with open(proc_dir / "ml_data.pkl",   "rb") as f:
        ml_data = pickle.load(f)
    with open(proc_dir / "lstm_data.pkl", "rb") as f:
        lstm_data = pickle.load(f)

    n_test     = len(ml_data["test_df"])
    test_dates = ml_data["test_df"].index

    # ── ARIMA predictions ─────────────────────────────────────────────────
    arima_model    = pickle.load(open(mdl_dir / "arima.pkl", "rb"))
    arima_forecast = arima_model.forecast(steps=n_test).values
    arima_true     = ml_data["y_test_raw"]

    # ── XGBoost predictions ───────────────────────────────────────────────
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(mdl_dir / "xgboost.json"))
    xgb_pred = inverse_scale(
        xgb_model.predict(ml_data["X_test"]), ml_data["target_scaler"]
    )
    xgb_true = ml_data["y_test_raw"]

    # ── LSTM predictions ──────────────────────────────────────────────────
    from tensorflow import keras
    lstm_model = keras.models.load_model(str(mdl_dir / "lstm.keras"))
    scaler     = lstm_data["scaler"]
    target_idx = lstm_data["target_idx"]
    n_features = len(lstm_data["all_cols"])
    lstm_dates = lstm_data["test_df"].index

    lstm_pred = _invert_lstm(
        lstm_model.predict(lstm_data["X_test"]).ravel(),
        scaler, target_idx, n_features,
    )
    lstm_true = _invert_lstm(
        lstm_data["y_test"], scaler, target_idx, n_features
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    for name, y_true, y_pred, dates in [
        ("ARIMA",   arima_true, arima_forecast, test_dates),
        ("XGBoost", xgb_true,   xgb_pred,       test_dates),
        ("LSTM",    lstm_true,  lstm_pred,       lstm_dates),
    ]:
        plot_forecast(
            y_true, y_pred, dates=dates,
            title=f"{name} — Forecast vs Actual",
            save_path=str(plots_dir / f"{name.lower()}_forecast.png"),
        )
        plot_residuals(
            y_true, y_pred,
            title=f"{name} — Residual Analysis",
            save_path=str(plots_dir / f"{name.lower()}_residuals.png"),
        )

    results = [
        compute_metrics(arima_true, arima_forecast, "ARIMA"),
        compute_metrics(xgb_true,   xgb_pred,       "XGBoost"),
        compute_metrics(lstm_true,  lstm_pred,       "LSTM"),
    ]
    plot_model_comparison(
        results,
        save_path=str(plots_dir / "model_comparison.png"),
    )

    print(f"Plots saved → {plots_dir.relative_to(ROOT)}/")
    for p in sorted(plots_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
