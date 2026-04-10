# Time-Series Forecasting — S&P 500 & Individual Stocks

An end-to-end machine-learning pipeline for stock price forecasting, covering data ingestion,
feature engineering, model training, hyperparameter tuning, and rich visualisation.
Three model families are compared side-by-side: classical ARIMA, gradient-boosted trees
(XGBoost), and deep learning (LSTM).

MLOps layer: experiments are tracked with **MLflow** on DagHub, data and models are
versioned with **DVC**, and a reproducible CLI pipeline (`dvc repro`) can retrain everything
from a single parameter change.

---

## Pipeline Overview

```
Yahoo Finance
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  data_load  │────▶│  preprocessing   │────▶│      eda         │
│  (raw OHLCV)│     │  (features +     │     │  (stats + plots) │
└─────────────┘     │   splits + scale)│     └──────────────────┘
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐   ┌──────────┐   ┌──────────┐
         │  ARIMA  │   │ XGBoost  │   │   LSTM   │
         └────┬────┘   └────┬─────┘   └────┬─────┘
              │             │              │
              └─────────────┴──────────────┘
                             │
                    ┌────────▼────────┐
                    │  fine_tuning    │
                    │ (grid / Optuna) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   evaluation    │
                    │ MAE/RMSE/MAPE/R²│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  visualization  │
                    │(static+Plotly)  │
                    └─────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd Time-Series-Forecasting-S-P-500

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Register the kernel so VS Code / Jupyter can find it
python -m ipykernel install --user --name=sp500-venv --display-name "Python (.venv)"

# 5. Create your credentials file (never committed)
#    Add these two lines:
#      DAGSHUB_USER=<your-dagshub-username>
#      DAGSHUB_TOKEN=<your-dagshub-token>
copy NUL .env      # Windows
# touch .env       # Mac/Linux

# 6. Pull data/models from DagHub (if someone already ran the pipeline)
.venv\Scripts\dvc pull
```

---

## User Configuration

All tunable settings live in **Cell 3** of the notebook (and in `params.yaml` for the CLI pipeline):

```python
TICKER     = 'AAPL'    # Any valid Yahoo Finance ticker:
                       #   Stocks:  'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'
                       #   Index:   '^GSPC' (S&P 500), '^IXIC' (NASDAQ), '^DJI'
                       #   Crypto:  'BTC-USD', 'ETH-USD'

YEARS      = 4         # Years of history to download (1–20)
SEQ_LEN    = 60        # LSTM look-back window in trading days (≈ 3 months)
TARGET_COL = 'Close'   # Column to forecast
```

---

## Project Structure

```
Time-Series-Forecasting-S-P-500/
│
├── notebooks/
│   ├── experiment.ipynb           ← Interactive ML notebook (no MLOps)
│   └── experiment_MLOps.ipynb     ← Same pipeline + MLflow tracking + DVC guide
│
├── src/
│   ├── data_load.py               ← Yahoo Finance downloader
│   ├── preprocessing.py           ← Feature engineering + splits + scaling
│   ├── eda.py                     ← Exploratory analysis + stationarity tests
│   ├── training.py                ← ARIMA / XGBoost / LSTM model builders
│   ├── fine_tuning.py             ← Grid search (ARIMA) + Optuna (XGB/LSTM)
│   ├── evaluation.py              ← Metrics: MAE, RMSE, MAPE, R², Dir.Acc
│   ├── visualization.py           ← Static (matplotlib) + interactive (Plotly)
│   └── pipeline/                  ← DVC stage scripts
│       ├── data_load.py           ← Stage 1: download → data/raw.csv
│       ├── preprocess.py          ← Stage 2: features → data/processed/
│       ├── train.py               ← Stage 3: train + MLflow log → models/
│       └── evaluate.py            ← Stage 4: plots → reports/plots/
│
├── params.yaml                    ← Single source of truth for all hyperparameters
├── dvc.yaml                       ← DVC pipeline definition (4 stages)
├── .env                           ← DagHub credentials (git-ignored)
├── requirements.txt
└── README.md
```

---

## MLOps: Two Workflows

### Workflow A — Interactive Notebook (learning / exploring)

Use `experiment_MLOps.ipynb` when you want to explore results interactively,
learn how the tracking works, or tweak code mid-run. Every training cell logs
automatically to MLflow on DagHub.

#### Step-by-step

**1. One-time setup**
```
Open VS Code
→ File > Open Folder > Time-Series-Forecasting-S-P-500
→ Open notebooks/experiment_MLOps.ipynb
→ Top-right kernel selector → choose "Python (.venv)"
```

**2. Fill in your credentials** (already done if you followed Quick Start)
```
.env file in project root must contain:
  DAGSHUB_USER=AmirhosseinnnKhademi
  DAGSHUB_TOKEN=<your-token>
```

**3. Run Section 0 — MLOps Setup** (cells 1–8)
- Loads `.env` credentials
- Connects MLflow to DagHub → prints a URL to your experiments page
- Verifies DVC remote is configured
- Sets `TICKER`, `YEARS`, `SEQ_LEN`

**4. Run Section 1 — Data Loading**
- Downloads OHLCV data and saves `data/raw.csv`
- The cell prints DVC commands you can optionally run in a terminal to version the file

**5. Run Sections 2–3 — EDA + Preprocessing**
- Standard feature engineering and train/val/test split
- No MLflow activity here

**6. Run Section 4 — Model Training** ← MLflow logs here
Each of the three training cells (`ARIMA`, `XGBoost`, `LSTM`) is wrapped in
`with mlflow.start_run():`. When the cell finishes, open DagHub to see the run:
```
https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500/experiments
```
You will see a new row per model with its params and metrics.

**7. Run Section 5 — Hyperparameter Tuning** ← MLflow logs nested runs here
- XGBoost: 20 Optuna trials, each logged as a child run inside `XGBoost-tuning`
- LSTM: same pattern inside `LSTM-tuning`
- In the DagHub UI, expand a parent run to compare every trial

**8. Run Section 6 — Final Evaluation**
- Tunes ARIMA (grid search), logs the tuned version
- Builds a leaderboard table and uploads it as a CSV artifact to MLflow

**9. Run Section 7 — Visualisation**
- Standard forecast, residual, and comparison plots (same as `experiment.ipynb`)

**10. Run Section 8 — DVC Pipeline info** (optional / reference)
- Shows `dvc status` so you can see which stages would re-run
- Explains the CLI workflow (Workflow B)

**11. Run Section 9 — Load a model from MLflow** (optional)
- Shows how to reload any saved model by run ID

---

### Workflow B — CLI Pipeline (reproducible / automated)

Use `dvc repro` when you want a fully reproducible run, want to change a
parameter and retrain only the affected stages, or want to automate training
in CI. No notebook needed.

#### Step-by-step

**1. Set credentials** (one-time, already done)
```
.env file contains DAGSHUB_USER and DAGSHUB_TOKEN
```

**2. Edit `params.yaml`** to change any parameter
```yaml
ticker: "MSFT"          # was AAPL
start_years_back: 5     # was 4
lstm:
  epochs: 150           # was 100
```

**3. Run the pipeline**
```bash
# From the project root:
.venv\Scripts\dvc repro
```
DVC compares file hashes. Only stages whose inputs changed are re-run.
For example, changing `ticker` reruns all 4 stages.
Changing only `lstm.epochs` reruns only `train` and `evaluate`.

Each stage prints its progress. The `train` stage logs all runs to MLflow
automatically (same as the notebook).

**4. Check what ran**
```bash
.venv\Scripts\dvc status      # shows which stages are up-to-date
.venv\Scripts\dvc dag         # prints the dependency graph
```

**5. Inspect metrics**
```bash
.venv\Scripts\dvc metrics show    # prints reports/metrics.json
.venv\Scripts\dvc metrics diff    # compares with previous run
```

**6. Push data and models to DagHub**
```bash
.venv\Scripts\dvc push
```
This uploads `data/`, `models/`, and `reports/` to DagHub remote storage.

**7. Commit the pipeline state to git**
```bash
git add dvc.lock params.yaml
git commit -m "retrain: MSFT 5yr"
git push
```
`dvc.lock` records the exact inputs and outputs of every stage.
Anyone who checks out this commit can run `dvc pull` + `dvc repro` to
reproduce the exact same results.

---

## Pushing to GitHub and DagHub

### What goes where

| Content | Where it lives | How to push |
|---------|---------------|-------------|
| Code, notebooks, params.yaml, dvc.yaml, dvc.lock, *.dvc files | GitHub (git) | `git push` |
| Raw data (data/raw.csv), processed data, models, plots | DagHub remote storage (DVC) | `dvc push` |
| Experiment runs (metrics, params, artifacts) | DagHub MLflow server | Automatic — logged during training |

### Step-by-step: push latest version

**Step 1 — Stage everything changed**
```bash
git add notebooks/experiment.ipynb
git add notebooks/experiment_MLOps.ipynb
git add src/
git add params.yaml dvc.yaml dvc.lock
git add requirements.txt README.md .gitignore
# Do NOT git add .env  (credentials — gitignored)
# Do NOT git add data/ models/ reports/  (DVC-tracked)
```

**Step 2 — Commit**
```bash
git commit -m "add MLOps layer: DVC pipeline + MLflow tracking"
```

**Step 3 — Push code to GitHub**
```bash
git push origin main
```
DagHub mirrors your GitHub repo automatically — no separate push needed for code.

**Step 4 — Push data/models to DagHub storage**
```bash
.venv\Scripts\dvc push
```
This uploads the actual CSV/model files to DagHub's S3-compatible storage.
The `.dvc` pointer files already committed in Step 1–3 now point to real content.

**Step 5 — Verify**
- GitHub: `https://github.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500`
- DagHub code mirror: `https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500`
- MLflow experiments: `https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500/experiments`
- DVC files: `https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500/src/branch/main/data`

### How someone else clones and reproduces

```bash
git clone https://github.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500
cd Time-Series-Forecasting-S-P-500
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Add credentials to .env
.venv\Scripts\dvc pull     # downloads data + models from DagHub
# Open experiment_MLOps.ipynb and run, OR:
.venv\Scripts\dvc repro    # reproduces all results from scratch
```

---

## Module Reference

### `src/data_load.py`

| Function | Description |
|---|---|
| `get_stock_data(delta, tickers)` | Core downloader — returns long-format DataFrame |
| `get_sp500_data(delta=5)` | Convenience wrapper for `^GSPC` |
| `get_tech_stocks(delta=5)` | Convenience wrapper for AAPL/MSFT/GOOGL/AMZN/NVDA |

### `src/preprocessing.py`

| Function | Description |
|---|---|
| `clean_data(df)` | Sort, deduplicate, forward-fill gaps |
| `add_technical_indicators(df)` | ~20 features: MAs, RSI, MACD, Bollinger Bands, returns, volatility |
| `add_lag_features(df, lags)` | Rolling lag features for the target column |
| `train_val_test_split(df, 0.70, 0.15)` | Strict chronological split |
| `prepare_ml_data(df, feature_cols)` | Full pipeline → dict with X/y arrays + scalers |
| `prepare_lstm_data(df, feature_cols, seq_len)` | Full pipeline including sequence creation |

**Generated features:**

| Group | Features |
|---|---|
| Moving averages | MA_7, MA_21, MA_50, MA_200, EMA_12, EMA_26 |
| Momentum | RSI_14, MACD, MACD_signal, MACD_hist |
| Volatility | BB_width, BB_pct, Volatility_20d |
| Returns | Return_1d, Return_5d, Return_20d |
| Volume | Volume_ratio (vs 20-day avg) |

### `src/training.py`

| Function | Description |
|---|---|
| `train_arima(series, order)` | Fit statsmodels ARIMA |
| `train_xgboost(X_train, y_train, X_val, y_val, params)` | XGBRegressor with early stopping |
| `build_lstm(input_shape, units, dropout, lr)` | Build stacked LSTM (Keras) |
| `train_lstm(model, X_train, y_train, X_val, y_val)` | Train with EarlyStopping + ReduceLROnPlateau |

### `src/fine_tuning.py`

| Function | Trials | Optimises |
|---|---|---|
| `tune_arima(series, p_values, d_values, q_values)` | Grid | AIC |
| `tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)` | Optuna (TPE) | Val RMSE |
| `tune_lstm(X_train, y_train, X_val, y_val, n_trials=20)` | Optuna (TPE) | Val MSE |

### `src/evaluation.py`

| Metric | Formula | Good value |
|---|---|---|
| **MAE** | mean \|actual − pred\| | As low as possible (USD) |
| **RMSE** | √(mean (actual − pred)²) | Lower than naive baseline |
| **MAPE** | mean \|actual − pred\| / actual × 100 | < 2 % is good for daily close |
| **R²** | 1 − SS_res / SS_tot | Closer to 1.0 |
| **Directional Accuracy** | % correct up/down calls | > 0.52 is meaningful |

### `src/visualization.py`

| Function | Description |
|---|---|
| `plot_forecast(y_true, y_pred, dates)` | Test-window: actual vs predicted + residual panel |
| `plot_forecast_with_history(...)` | Full history with train/val/test bands + split lines |
| `plot_model_comparison(results)` | Bar charts comparing all models across all metrics |
| `plot_loss_curves(history)` | LSTM train/val loss and MAE per epoch |
| `plot_feature_importance(model, feature_names)` | XGBoost feature importance (top-N) |
| `plot_residuals(y_true, y_pred)` | 4-panel residual diagnostic |
| `plot_candlestick(df, ticker, last_n_days)` | Interactive Plotly OHLCV candlestick + volume |
| `plot_forecast_interactive(y_true, y_pred, dates)` | Interactive Plotly actual vs predicted |

---

## What to Tune

### ARIMA
Expand the grid search in notebook Section 5a or in `params.yaml`:
```python
tune_arima(train_series, p_values=[0,1,2,3], d_values=[0,1], q_values=[0,1,2,3])
```

### XGBoost
Increase `n_trials` or widen the search space in `fine_tuning.py → tune_xgboost`.
Also consider adding more lag features:
```python
add_lag_features(df, lags=[1, 2, 3, 5, 10, 20, 60])
```

### LSTM
Increase `n_trials` and `max_epochs`:
```python
tune_lstm(..., n_trials=40, max_epochs=100)
```
Or extend `SEQ_LEN` to 120 days to capture longer cycles.

### More data
Set `YEARS = 10` in the config cell or in `params.yaml`. More data generally
helps LSTM and XGBoost but makes Optuna tuning slower.

---

## Dependencies

```
pandas>=2.0        numpy>=1.24         matplotlib>=3.7
seaborn>=0.12      yfinance>=0.2       scikit-learn>=1.3
statsmodels>=0.14  xgboost>=2.0        tensorflow>=2.13
optuna>=3.3        plotly>=5.18        nbformat>=4.2.0

# MLOps
mlflow>=2.10       dvc>=3.0            dvc-s3>=3.0
dagshub>=0.3       python-dotenv>=1.0  pyyaml>=6.0
```

```bash
pip install -r requirements.txt
```

---

## Notes

- **TF warnings**: `Could not find cuda drivers` is harmless — CPU fallback is automatic.
- **Interactive charts**: require `nbformat>=4.2.0`. Run `pip install nbformat` and restart kernel if blank.
- **Kernel not found**: run `python -m ipykernel install --user --name=sp500-venv` then reload VS Code.
- **yfinance rate limits**: if a download fails, wait a few seconds and retry.
- **DVC credentials**: stored in `.dvc/config.local` (auto-gitignored) and `.env` — never committed.
- **MLflow runs offline**: if DagHub is unreachable, MLflow falls back to a local `mlruns/` folder.
