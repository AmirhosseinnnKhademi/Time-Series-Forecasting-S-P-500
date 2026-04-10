# Project Reimplementation Guide — A to Z

Everything you need to reproduce this project from a fresh clone, including every file,
terminal command, and decision point.

---

## 0 · Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| Git | any recent |
| Internet access | for data download + DagHub |

---

## 1 · Clone the Repository

```bash
git clone https://github.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500.git
cd Time-Series-Forecasting-S-P-500
```

---

## 2 · Create and Activate the Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (Windows cmd/PowerShell)
.venv\Scripts\activate

# You should now see (.venv) at the start of your terminal prompt
```

---

## 3 · Install All Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:
- `yfinance` — downloads stock data
- `statsmodels` — ARIMA model
- `xgboost` — gradient boosting model
- `tensorflow` / `keras` — LSTM model
- `scikit-learn` — scaling, metrics
- `optuna` — hyperparameter tuning
- `mlflow` — experiment tracking
- `dvc`, `dvc-s3` — data versioning
- `dagshub` — remote MLflow + DVC storage
- `python-dotenv` — load `.env` credentials
- `plotly` — interactive charts
- `nbformat` — notebook utilities

---

## 4 · Set Up Credentials

Create a `.env` file in the project root (this file is gitignored — never commit it):

```
# .env
DAGSHUB_USER=AmirhosseinnnKhademi
DAGSHUB_TOKEN=<your-dagshub-token>
```

Get your token from: DagHub → Settings → Tokens

---

## 5 · Configure DVC Remote (one-time setup)

DVC is already initialised (`.dvc/config` is in the repo). Add credentials locally:

```bash
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user AmirhosseinnnKhademi
dvc remote modify --local dagshub password <your-dagshub-token>
```

This writes to `.dvc/config.local` which is gitignored automatically.

---

## 6 · Pull Data and Models from DVC Remote

Instead of re-downloading and re-training from scratch, pull what was already pushed:

```bash
dvc pull
```

This downloads:
- `data/raw.csv` — raw OHLCV stock data
- `data/processed/ml_data.pkl` — flat feature matrices for XGBoost/ARIMA
- `data/processed/lstm_data.pkl` — 3D tensors for LSTM
- `models/arima.pkl` — trained ARIMA model
- `models/xgboost.json` — trained XGBoost model
- `models/lstm.keras` — trained LSTM model
- `reports/plots/*.png` — evaluation charts

If you want to retrain from scratch instead, skip this step and follow Section 8 (CLI) or Section 9 (Notebook).

---

## 7 · Key Configuration Files

### `params.yaml` — single source of truth for all hyperparameters

```yaml
ticker: "AAPL"          # stock ticker symbol
start_years_back: 4     # how many years of history to download
target_col: "Close"     # column to predict
train_ratio: 0.70       # 70% of data for training
val_ratio: 0.15         # 15% for validation, remaining 15% for test
seq_len: 60             # LSTM lookback window (trading days)

arima:
  p: 1                  # autoregressive order
  d: 1                  # differencing order
  q: 1                  # moving average order

xgboost:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 3
  reg_alpha: 0.01
  reg_lambda: 1.0

lstm:
  units_1: 64
  units_2: 32
  dropout: 0.2
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
  patience: 15
```

**To change the stock**: edit `ticker`. To change model behaviour: edit the relevant block. After any edit, re-run the pipeline (Section 8).

### `dvc.yaml` — pipeline stage definitions (do not edit manually)

Defines the 4-stage pipeline and what each stage depends on, reads, and writes.

### `.env` — credentials (never commit, see Section 4)

### `requirements.txt` — all Python dependencies

---

## 8 · Workflow A — CLI Pipeline (reproducible, DVC-tracked)

Use this when you want a fully reproducible run tracked by DVC and MLflow.

### Step 1: Edit hyperparameters (optional)
```
edit params.yaml
```

### Step 2: Run the full pipeline
```bash
dvc repro
```

DVC runs only stages whose inputs changed:

| Stage | Script | Input → Output |
|---|---|---|
| `data_load` | `src/pipeline/data_load.py` | `params.yaml` → `data/raw.csv` |
| `preprocess` | `src/pipeline/preprocess.py` | `data/raw.csv` → `data/processed/` |
| `train` | `src/pipeline/train.py` | `data/processed/` → `models/` + `reports/metrics.json` |
| `evaluate` | `src/pipeline/evaluate.py` | `models/` → `reports/plots/` |

### Step 3: Check metrics
```bash
dvc metrics show
```
Shows MAE, RMSE, MAPE, R², Dir.Acc for ARIMA, XGBoost, LSTM.

### Step 4: Compare across runs
```bash
dvc metrics diff
```

### Step 5: Push data + models to DagHub
```bash
dvc push
```

### Step 6: Commit and push to GitHub
```bash
git add dvc.lock reports/metrics.json
git commit -m "retrain: <describe what changed>"
git push origin main
```

---

## 9 · Workflow B — Notebook (interactive, exploratory)

Use this for EDA, experimentation, Optuna tuning, and visualisation.

### Step 1: Open VS Code and activate the venv kernel
```
Open VS Code → open notebooks/experiment_MLOps.ipynb
Select kernel → .venv (Python 3.x)
```

### Step 2: Revert the notebook to get the latest saved version
```
File → Revert File
```
Always do this if the file was edited externally (e.g. by a script).

### Step 3: Restart the kernel
```
Kernel → Restart Kernel
```

### Step 4: Run sections in order

| Section | What it does |
|---|---|
| **0 — MLOps Setup** | Loads credentials, connects MLflow to DagHub, verifies DVC |
| **User Config** | Set ticker, years, seq_len, train/val ratios |
| **1 — Data Loading** | Downloads OHLCV data via yfinance, saves `data/raw.csv` |
| **2 — EDA** | Price history, returns distribution, rolling stats, ACF/PACF, decomposition |
| **3 — Preprocessing** | Feature engineering, train/val/test split, scaling |
| **4 — Training + MLflow** | Trains ARIMA, XGBoost, LSTM with default params; logs to DagHub |
| **5 — Optuna Tuning** | 5a: ARIMA grid search, 5b: XGBoost Optuna, 5c: LSTM Optuna |
| **6 — Final Evaluation** | Leaderboard comparing all 6 models (default + tuned) |
| **7 — Visualisation** | All plots: forecast, residuals, comparison, candlestick, interactive |
| **8 — DVC Pipeline** | Reference guide for CLI workflow |
| **9 — MLflow Reload** | Load any saved model from MLflow by run ID |

### Step 5: After tuning — write best params to `params.yaml`
Take the best params printed at the end of sections 5a/5b/5c and manually update `params.yaml`. Then run `dvc repro` (Workflow A) for the reproducible pipeline run.

### Step 6: Commit notebook outputs
```bash
git add notebooks/experiment_MLOps.ipynb
git commit -m "notebook: completed tuning run"
git push origin main
```

---

## 10 · Source File Map

```
src/
├── data_load.py          # get_stock_data() — downloads OHLCV via yfinance
├── preprocessing.py      # clean_data(), add_technical_indicators(),
│                         # add_lag_features(), prepare_ml_data(),
│                         # prepare_lstm_data()
├── eda.py                # summary_stats(), plot_price_history(),
│                         # plot_returns_distribution(), plot_rolling_stats(),
│                         # plot_acf_pacf(), plot_decomposition(),
│                         # plot_correlation_heatmap()
├── training.py           # train_arima(), train_xgboost(),
│                         # build_lstm(), train_lstm()
├── evaluation.py         # compute_metrics(), compare_models(),
│                         # directional_accuracy(), inverse_scale()
├── fine_tuning.py        # tune_arima(), tune_xgboost(), tune_lstm()
├── visualization.py      # plot_forecast(), plot_forecast_with_history(),
│                         # plot_model_comparison(), plot_loss_curves(),
│                         # plot_feature_importance(), plot_residuals(),
│                         # plot_candlestick(), plot_forecast_interactive()
└── pipeline/
    ├── data_load.py      # DVC Stage 1 — calls src/data_load.py
    ├── preprocess.py     # DVC Stage 2 — calls src/preprocessing.py
    ├── train.py          # DVC Stage 3 — calls src/training.py + MLflow
    └── evaluate.py       # DVC Stage 4 — calls src/visualization.py
```

**Rule**: `src/*.py` = reusable library modules. `src/pipeline/*.py` = thin CLI wrappers that read `params.yaml`, call the library, and write outputs.

---

## 11 · DVC Command Reference

| Command | What it does |
|---|---|
| `dvc repro` | Re-run only changed stages |
| `dvc repro -f <stage>` | Force re-run a specific stage |
| `dvc push` | Upload cached data/models to DagHub |
| `dvc pull` | Download data/models from DagHub |
| `dvc status` | Show which stages are out of date |
| `dvc metrics show` | Print metrics from `reports/metrics.json` |
| `dvc metrics diff` | Compare metrics between git commits |
| `dvc dag` | Print the pipeline dependency graph |

---

## 12 · MLflow / DagHub Reference

All experiments are visible at:
```
https://dagshub.com/AmirhosseinnnKhademi/Time-Series-Forecasting-S-P-500/experiments
```

| Experiment name | Created by |
|---|---|
| `SP500-Forecasting-Notebook` | `experiment_MLOps.ipynb` sections 4–5 |
| `SP500-DVC-Pipeline` | `src/pipeline/train.py` via `dvc repro` |

To reload a model from MLflow by run ID (in the notebook):
```python
import mlflow
model = mlflow.xgboost.load_model(f'runs:/<run_id>/model')
```

---

## 13 · Full Re-implementation Checklist

```
[ ] git clone
[ ] python -m venv .venv && .venv\Scripts\activate
[ ] pip install -r requirements.txt
[ ] Create .env with DAGSHUB_USER and DAGSHUB_TOKEN
[ ] dvc remote modify --local dagshub auth basic
[ ] dvc remote modify --local dagshub user <user>
[ ] dvc remote modify --local dagshub password <token>
[ ] dvc pull              ← get data + models (or skip to retrain)
[ ] Edit params.yaml      ← change ticker / hyperparams if needed
[ ] dvc repro             ← run CLI pipeline (Workflow A)
    OR
[ ] Run experiment_MLOps.ipynb sections 0–7  (Workflow B)
[ ] dvc push              ← upload new artifacts
[ ] git add dvc.lock reports/metrics.json
[ ] git commit -m "..."
[ ] git push origin main
```
