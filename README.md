# Time-Series Forecasting — S&P 500 & Individual Stocks

An end-to-end machine-learning pipeline for stock price forecasting, covering data ingestion, feature engineering, model training, hyperparameter tuning, and rich visualisation. Three model families are compared side-by-side: classical ARIMA, gradient-boosted trees (XGBoost), and deep learning (LSTM).

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
              └──────────────┴──────────────┘
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
                    │ visualization   │
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

# 5. Open the notebook
code notebooks/experiment.ipynb
# Select kernel: "Python (.venv)"
```

---

## User Configuration

All tunable settings live in **Cell 3** of the notebook (the "User Configuration" cell):

```python
TICKER     = 'AAPL'    # Any valid Yahoo Finance ticker:
                       #   Stocks:  'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'
                       #   Index:   '^GSPC' (S&P 500), '^IXIC' (NASDAQ), '^DJI'
                       #   Crypto:  'BTC-USD', 'ETH-USD'

YEARS      = 5         # How many years of history to download (1–20)
                       #   More years = more training data but slower fine-tuning

SEQ_LEN    = 60        # LSTM look-back window in trading days (default: 60 ≈ 3 months)
                       #   Shorter (30): captures recent trends, faster training
                       #   Longer (120): captures longer cycles, more memory needed

TARGET_COL = 'Close'   # Column to forecast — 'Close' is standard
```

Change these values and **Run All** to get a fresh analysis for any ticker.

---

## Project Structure

```
Time-Series-Forecasting-S-P-500/
├── notebooks/
│   └── experiment.ipynb       ← Main notebook (run this)
├── src/
│   ├── data_load.py           ← Yahoo Finance downloader
│   ├── preprocessing.py       ← Feature engineering + splits + scaling
│   ├── eda.py                 ← Exploratory analysis + stationarity tests
│   ├── training.py            ← ARIMA / XGBoost / LSTM model builders
│   ├── fine_tuning.py         ← Grid search (ARIMA) + Optuna (XGB/LSTM)
│   ├── evaluation.py          ← Metrics: MAE, RMSE, MAPE, R², Dir.Acc
│   └── visualization.py       ← Static (matplotlib) + interactive (Plotly) charts
├── requirements.txt
└── README.md
```

---

## Module Reference

### `src/data_load.py`

Downloads OHLCV data via `yfinance`.

| Function | Description |
|---|---|
| `get_stock_data(delta, tickers)` | Core downloader — returns long-format DataFrame with a `ticker` column |
| `get_sp500_data(delta=5)` | Convenience wrapper for `^GSPC` |
| `get_tech_stocks(delta=5)` | Convenience wrapper for AAPL/MSFT/GOOGL/AMZN/NVDA basket |

### `src/preprocessing.py`

| Function | Description |
|---|---|
| `clean_data(df)` | Sort, deduplicate, forward-fill gaps |
| `add_technical_indicators(df)` | Adds ~20 features (MAs, RSI, MACD, Bollinger Bands, returns, volatility) |
| `add_lag_features(df, lags)` | Rolling lag features for the target column |
| `train_val_test_split(df, 0.70, 0.15)` | Strict chronological split |
| `scale_features(X_train, X_val, X_test)` | MinMaxScaler fitted on train only |
| `create_sequences(X, y, seq_len)` | Rolling windows → 3-D tensor for LSTM |
| `prepare_ml_data(df, feature_cols)` | Full pipeline → dict with X/y arrays + scalers |
| `prepare_lstm_data(df, feature_cols, seq_len)` | Full pipeline including sequence creation |

**Generated features:**

| Group | Features |
|---|---|
| Moving averages | MA_7, MA_21, MA_50, MA_200, EMA_12, EMA_26 |
| Momentum | RSI_14, MACD, MACD_signal, MACD_hist |
| Volatility | BB_width, BB_pct, Volatility_20 |
| Returns | Return_1d, Return_5d, Return_20d |
| Volume | Volume_ratio (vs 20-day avg) |

### `src/eda.py`

| Function | Description |
|---|---|
| `summary_stats(df)` | Descriptive statistics |
| `returns_summary(df)` | Annualised return/vol, Sharpe ratio, max drawdown |
| `adf_test(series)` / `kpss_test(series)` | Stationarity tests |
| `plot_price_history(df)` | Price + MA overlays + volume |
| `plot_returns_distribution(df)` | Histogram + KDE + Q-Q plot |
| `plot_rolling_stats(series)` | Rolling mean and std |
| `plot_acf_pacf(series, lags)` | ACF and PACF — guides ARIMA order selection |
| `plot_decomposition(series, period)` | Trend + seasonal + residual components |
| `plot_correlation_heatmap(df)` | Feature correlation matrix |

### `src/training.py`

| Function | Description |
|---|---|
| `train_arima(series, order)` | Fit statsmodels ARIMA |
| `arima_rolling_forecast(series, train_size, order)` | Walk-forward one-step-ahead forecasts |
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
| `plot_forecast_with_history(full_series, y_true, y_pred, train_end, val_end)` | **Full history** with train/val/test bands + dashed split lines |
| `plot_model_comparison(results)` | Bar charts comparing all models across all metrics |
| `plot_loss_curves(history)` | LSTM train/val loss and MAE per epoch |
| `plot_feature_importance(model, feature_names)` | XGBoost feature importance (top-N) |
| `plot_residuals(y_true, y_pred)` | 4-panel: time/histogram/pred-vs-actual/residuals-vs-pred |
| `plot_candlestick(df, ticker, last_n_days)` | Interactive Plotly OHLCV candlestick + volume |
| `plot_forecast_interactive(y_true, y_pred, dates)` | Interactive Plotly actual vs predicted |

---

## What to Tune

### To improve ARIMA

In the notebook's **Section 5a**, expand the grid search:
```python
best_arima_order, _ = tune_arima(
    train_series,
    p_values=[0, 1, 2, 3],
    d_values=[0, 1, 2],
    q_values=[0, 1, 2, 3],
)
```
More (p, d, q) combinations → longer search but better AIC. For seasonal data (e.g. monthly), consider SARIMA.

### To improve XGBoost

In **Section 5b**, increase `n_trials`:
```python
best_xgb_params, _ = tune_xgboost(..., n_trials=100)  # default 50
```
Also consider adding more lag features in `preprocessing.py`:
```python
add_lag_features(df, lags=[1, 2, 3, 5, 10, 20, 60])
```

### To improve LSTM

In **Section 5c**, increase `n_trials` and `max_epochs`:
```python
best_lstm_params, _ = tune_lstm(..., n_trials=40, max_epochs=100)
```
Or widen the search space in `fine_tuning.py → tune_lstm → objective`:
```python
units_1 = trial.suggest_categorical("units_1", [64, 128, 256])
units_2 = trial.suggest_categorical("units_2", [32, 64, 128])
```

### To use more data

Change `YEARS = 10` in the User Configuration cell. More data generally helps LSTM and XGBoost. For very long histories, consider using only the last N years for validation/test to keep the split periods meaningful.

### To forecast a different horizon

The current setup forecasts **one step ahead** (next trading day). To forecast N days ahead:
1. In `preprocessing.py → create_sequences`, shift `y` by `N` instead of 1
2. In `training.py → build_lstm`, change the final `Dense(1)` to `Dense(N)` and adjust loss
3. For ARIMA, `model.forecast(steps=N)` already supports multi-step

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
yfinance>=0.2
scikit-learn>=1.3
statsmodels>=0.14
xgboost>=2.0
tensorflow>=2.13
optuna>=3.3
plotly>=5.18
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Notes

- **TensorFlow warnings**: Messages like `Could not find cuda drivers` are harmless — TensorFlow falls back to CPU automatically. They are suppressed in the notebook's setup cell.
- **Interactive charts**: Plotly charts require `nbformat` to render inline in VS Code. If blank, run `pip install plotly nbformat` and restart the kernel.
- **Kernel not found**: Run `.venv/bin/python -m ipykernel install --user --name=sp500-venv --display-name "Python (.venv)"` then reload VS Code.
- **yfinance data**: Free API, subject to rate limits. If a download fails, wait a few seconds and retry.
