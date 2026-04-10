"""DVC Stage 2 — Feature engineering and train/val/test split.

Reads  : data/raw.csv
Writes : data/processed/ml_data.pkl   (flat matrices for XGBoost / ARIMA)
         data/processed/lstm_data.pkl  (3-D tensors for LSTM)
         data/processed/feature_cols.txt

Run directly:   python src/pipeline/preprocess.py
Run via DVC:    dvc repro preprocess
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    add_lag_features,
    add_technical_indicators,
    clean_data,
    prepare_lstm_data,
    prepare_ml_data,
)


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

    target_col  = params["target_col"]
    seq_len     = params["seq_len"]
    train_ratio = params["train_ratio"]
    val_ratio   = params["val_ratio"]

    # ── Load raw data ──────────────────────────────────────────────────────
    raw = pd.read_csv(ROOT / "data" / "raw.csv", index_col=0, parse_dates=True)

    # ── Feature engineering ────────────────────────────────────────────────
    df = clean_data(raw)
    df = add_technical_indicators(df, close_col=target_col)
    df = add_lag_features(df, col=target_col)

    # Feature columns = everything except the raw price/volume columns that
    # are already captured by the technical indicators.
    exclude = {"Open", "High", "Low", "Volume"}
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"Features: {len(feature_cols)}  |  Rows after dropna: ", end="")

    # ── Build data splits ──────────────────────────────────────────────────
    ml_data   = prepare_ml_data(df, feature_cols, target_col=target_col,
                                train_ratio=train_ratio, val_ratio=val_ratio)
    lstm_data = prepare_lstm_data(df, feature_cols, target_col=target_col,
                                  seq_len=seq_len,
                                  train_ratio=train_ratio, val_ratio=val_ratio)

    n_train = len(ml_data["train_df"])
    n_val   = len(ml_data["val_df"])
    n_test  = len(ml_data["test_df"])
    print(f"{n_train + n_val + n_test}")
    print(f"  Train {n_train} | Val {n_val} | Test {n_test}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ml_data.pkl",   "wb") as f:
        pickle.dump(ml_data, f)
    with open(out_dir / "lstm_data.pkl", "wb") as f:
        pickle.dump(lstm_data, f)
    (out_dir / "feature_cols.txt").write_text("\n".join(feature_cols))

    print(f"Saved processed data → {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
