"""DVC Stage 3 — Train ARIMA, XGBoost, and LSTM; log everything to MLflow.

Reads  : data/processed/ml_data.pkl
         data/processed/lstm_data.pkl
         params.yaml
Writes : models/arima.pkl
         models/xgboost.json
         models/lstm.keras      (Keras native format)
         reports/metrics.json   (DVC metrics file)

Run directly:   python src/pipeline/train.py
Run via DVC:    dvc repro train
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.evaluation import compute_metrics, inverse_scale  # noqa: E402
from src.training import (  # noqa: E402
    build_lstm,
    train_arima,
    train_lstm,
    train_xgboost,
)

warnings.filterwarnings("ignore")

# ── MLflow setup ────────────────────────────────────────────────────────────
import mlflow
import mlflow.keras
import mlflow.xgboost

MLFLOW_URI = "https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500.mlflow"
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.getenv("DAGSHUB_USER", ""))
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("SP500-DVC-Pipeline")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _log_metrics(metrics: dict) -> None:
    """Log a compute_metrics dict to the active MLflow run (skip non-numeric)."""
    mlflow.log_metrics({
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and k != "Model"
    })


def _invert_lstm(pred_scaled, scaler, target_idx, n_features):
    dummy = np.zeros((len(pred_scaled), n_features))
    dummy[:, target_idx] = pred_scaled
    return scaler.inverse_transform(dummy)[:, target_idx]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

    # Load processed data
    proc = ROOT / "data" / "processed"
    with open(proc / "ml_data.pkl",   "rb") as f:
        ml_data = pickle.load(f)
    with open(proc / "lstm_data.pkl", "rb") as f:
        lstm_data = pickle.load(f)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    n_test = len(ml_data["test_df"])
    all_results: list[dict] = []

    # ── ARIMA ────────────────────────────────────────────────────────────────
    print("\n[1/3] Training ARIMA …")
    arima_p = params["arima"]
    order   = (arima_p["p"], arima_p["d"], arima_p["q"])

    with mlflow.start_run(run_name="ARIMA"):
        mlflow.log_params({"p": order[0], "d": order[1], "q": order[2]})
        mlflow.set_tag("model_type", "statistical")

        arima_model    = train_arima(ml_data["train_df"]["Close"], order=order)
        arima_forecast = arima_model.forecast(steps=n_test).values
        arima_true     = ml_data["y_test_raw"]

        m = compute_metrics(arima_true, arima_forecast, "ARIMA")
        _log_metrics(m)
        all_results.append(m)

        pickle.dump(arima_model, open(models_dir / "arima.pkl", "wb"))
        mlflow.log_artifact(str(models_dir / "arima.pkl"), artifact_path="model")

    print(f"  ARIMA done — RMSE={m['RMSE']:.4f}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training XGBoost …")
    xgb_p = params["xgboost"]

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params(xgb_p)
        mlflow.set_tag("model_type", "tree")

        xgb_model = train_xgboost(
            ml_data["X_train"], ml_data["y_train"],
            ml_data["X_val"],   ml_data["y_val"],
            params=xgb_p,
        )
        xgb_pred = inverse_scale(
            xgb_model.predict(ml_data["X_test"]), ml_data["target_scaler"]
        )
        xgb_true = ml_data["y_test_raw"]

        m = compute_metrics(xgb_true, xgb_pred, "XGBoost")
        _log_metrics(m)
        all_results.append(m)

        mlflow.xgboost.log_model(xgb_model, artifact_path="model")
        xgb_model.save_model(str(models_dir / "xgboost.json"))

    print(f"  XGBoost done — RMSE={m['RMSE']:.4f}")

    # ── LSTM ─────────────────────────────────────────────────────────────────
    print("\n[3/3] Training LSTM …")
    lstm_p      = params["lstm"]
    input_shape = lstm_data["X_train"].shape[1:]
    scaler      = lstm_data["scaler"]
    target_idx  = lstm_data["target_idx"]
    n_features  = len(lstm_data["all_cols"])

    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_params({
            "units_1":       lstm_p["units_1"],
            "units_2":       lstm_p["units_2"],
            "dropout":       lstm_p["dropout"],
            "learning_rate": lstm_p["learning_rate"],
            "seq_len":       params["seq_len"],
            "batch_size":    lstm_p["batch_size"],
            "patience":      lstm_p["patience"],
        })
        mlflow.set_tag("model_type", "deep_learning")

        lstm_model = build_lstm(
            input_shape=input_shape,
            units=(lstm_p["units_1"], lstm_p["units_2"]),
            dropout=lstm_p["dropout"],
            learning_rate=lstm_p["learning_rate"],
        )
        lstm_model, history = train_lstm(
            lstm_model,
            lstm_data["X_train"], lstm_data["y_train"],
            lstm_data["X_val"],   lstm_data["y_val"],
            epochs=lstm_p["epochs"],
            batch_size=lstm_p["batch_size"],
            patience=lstm_p["patience"],
        )

        # Log per-epoch metrics so you can visualise curves in MLflow UI
        for epoch, (tl, vl) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metric("train_loss", tl, step=epoch)
            mlflow.log_metric("val_loss",   vl, step=epoch)

        lstm_pred = _invert_lstm(
            lstm_model.predict(lstm_data["X_test"]).ravel(),
            scaler, target_idx, n_features,
        )
        lstm_true = _invert_lstm(
            lstm_data["y_test"], scaler, target_idx, n_features
        )

        m = compute_metrics(lstm_true, lstm_pred, "LSTM")
        _log_metrics(m)
        all_results.append(m)

        mlflow.keras.log_model(lstm_model, artifact_path="model")
        lstm_model.save(str(models_dir / "lstm.keras"))

    print(f"  LSTM done — RMSE={m['RMSE']:.4f}")

    # ── Save DVC metrics file ─────────────────────────────────────────────────
    metrics_out = {r["Model"]: {k: v for k, v in r.items() if k != "Model"}
                   for r in all_results}
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2)
    )
    print(f"\nMetrics saved → reports/metrics.json")
    print("\nAll runs logged to MLflow:")
    print(f"  {MLFLOW_URI}")


if __name__ == "__main__":
    main()
