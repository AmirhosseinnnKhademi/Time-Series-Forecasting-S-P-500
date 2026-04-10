"""DVC Stage 1 — Download raw OHLCV data and save to data/raw.csv.

Run directly:   python src/pipeline/data_load.py
Run via DVC:    dvc repro data_load
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Make the project root importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_load import get_stock_data  # noqa: E402


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

    ticker = params["ticker"]
    years  = params["start_years_back"]

    print(f"Downloading {ticker} — {years} years of history …")
    df = get_stock_data(years, [ticker])

    # Drop the helper 'ticker' column if present (single-ticker run)
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    out_path = ROOT / "data" / "raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved {len(df):,} rows → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
