# Theoretical Background

The "why" behind every design decision in this project — preprocessing, features,
models, metrics, and MLOps tooling.

---

## 1 · Why Preprocessing?

Raw stock data cannot be fed directly into most models. Three problems must be solved first:

### 1.1 Missing values and market gaps
Stock exchanges are closed on weekends and holidays. The raw OHLCV series has gaps.
`clean_data()` forward-fills these so the series is contiguous and models see a uniform
time step.

### 1.2 Non-stationarity
Most statistical and ML models assume the data's statistical properties (mean, variance)
are constant over time. Stock prices are non-stationary — they trend upward over decades.
- **ARIMA** handles this explicitly with the `d` (differencing) parameter. `d=1` means
  the model is fit on `price[t] - price[t-1]` (returns), which is stationary.
- **XGBoost / LSTM** handle it implicitly through features like returns and technical
  indicators that are inherently stationary (RSI is bounded 0–100, returns oscillate
  around zero).

### 1.3 Scale differences
Features span wildly different ranges: Close price (~$100–$500), Volume (~millions),
RSI (0–100), MACD (small decimals). Neural networks and distance-based models are
sensitive to scale. `MinMaxScaler` compresses all features to [0, 1], so no single
feature dominates the gradient updates during LSTM training.

---

## 2 · Why EDA? What to Expect from Each Plot

Exploratory Data Analysis is not optional — it tells you whether your modelling
assumptions are valid before you spend hours training.

### 2.1 Price History (`plot_price_history`)
**What you see**: The raw close price over time.
**What to look for**:
- Long-term trend (expected — stock prices grow)
- Structural breaks (COVID crash 2020, rate hike 2022) — these challenge all models
- Regime changes: low-volatility vs high-volatility periods

### 2.2 Returns Distribution (`plot_returns_distribution`)
**What you see**: Histogram of daily percentage returns.
**What to look for**:
- Should be approximately bell-shaped but with **fat tails** (leptokurtic) — extreme
  moves happen more often than a normal distribution predicts
- Near-zero mean (markets are roughly a random walk on short time scales)
- Negative skew is common (crashes are faster than rallies)

### 2.3 Rolling Statistics (`plot_rolling_stats`)
**What you see**: 20-day and 50-day rolling mean and standard deviation.
**What to look for**:
- Rolling mean drifts → confirms non-stationarity → justifies differencing in ARIMA
- Rolling std spikes during crises → volatility clustering (calm periods followed by
  turbulent periods) → justifies the `Volatility_20d` feature

### 2.4 ACF / PACF (`plot_acf_pacf`)
**What you see**: Autocorrelation and partial autocorrelation of returns at different lags.
**What to look for**:
- If ACF dies off quickly and PACF cuts off at lag p → AR(p) process → guides ARIMA `p`
- If PACF dies off and ACF cuts at lag q → MA(q) process → guides ARIMA `q`
- Near-zero autocorrelations at all lags → returns are close to white noise (efficient
  market hypothesis) → ARIMA will struggle, motivating the use of XGBoost/LSTM with
  richer features

### 2.5 Seasonal Decomposition (`plot_decomposition`)
**What you see**: Trend, seasonal, and residual components separated.
**What to look for**:
- Trend component: confirms the long-run direction
- Seasonal component: weekly/monthly patterns (usually weak in stocks)
- Residual: should look like white noise if decomposition captured all structure

### 2.6 Correlation Heatmap (`plot_correlation_heatmap`)
**What you see**: Pairwise Pearson correlations between all features.
**What to look for**:
- Highly correlated features (> 0.9) are redundant — XGBoost handles this but LSTM
  is more sensitive
- MA_7, MA_21, MA_50 are all highly correlated with Close (by construction)
- RSI and returns are moderately correlated (RSI measures cumulative recent returns)
- Weak correlation with target may still be useful to tree models through interactions

---

## 3 · Why These Features? (Technical Indicators)

All features are derived from price and volume — no external data. This is deliberate:
the model must work with only what the market itself reveals.

### 3.1 Simple Moving Averages — MA_7, MA_21, MA_50

```
MA_w[t] = mean(Close[t-w+1], ..., Close[t])
```

**Why**: Moving averages smooth out noise and reveal trend direction. The crossover of
a short MA above a long MA (e.g. MA_7 > MA_50) is the classic "golden cross" buy signal.

| Window | Represents |
|---|---|
| MA_7 | 1.5-week trend — very short-term momentum |
| MA_21 | ~1-month trend — medium-term direction |
| MA_50 | ~2.5-month trend — institutional trend-following |

**As model features**: The ratio `Close / MA_50` tells the model how far price has
deviated from its medium-term average — a mean-reversion signal.

### 3.2 Exponential Moving Averages — EMA_12, EMA_26

```
EMA[t] = α × Close[t] + (1 − α) × EMA[t−1],   α = 2 / (span + 1)
```

**Why**: EMA weights recent prices more heavily than older ones (exponential decay).
This makes it more responsive to recent price changes than a simple MA.
EMA_12 and EMA_26 are the building blocks of MACD.

### 3.3 MACD — MACD, MACD_signal, MACD_hist

```
MACD        = EMA_12 − EMA_26
MACD_signal = EMA_9(MACD)
MACD_hist   = MACD − MACD_signal
```

**Why**: MACD measures the difference between two momentum estimates at different speeds.
- Positive MACD → short-term momentum stronger than long-term → bullish
- MACD crossing above MACD_signal → classic buy signal
- Histogram shows the acceleration/deceleration of momentum

### 3.4 RSI_14 — Relative Strength Index

```
RSI = 100 − 100 / (1 + RS),   RS = avg_gain_14d / avg_loss_14d
```

**Why**: RSI is a bounded (0–100) momentum oscillator.
- RSI > 70 → overbought (price moved up too fast, potential reversal)
- RSI < 30 → oversold (price moved down too fast, potential bounce)
- RSI ≈ 50 → neutral momentum

RSI is valuable because it is **stationary by construction** (always 0–100) unlike raw price.

### 3.5 Bollinger Bands — BB_width, BB_pct

```
SMA_20  = 20-day simple moving average
std_20  = 20-day rolling standard deviation
BB_upper = SMA_20 + 2 × std_20
BB_lower = SMA_20 − 2 × std_20
BB_width = (BB_upper − BB_lower) / SMA_20    ← volatility measure
BB_pct   = (Close − BB_lower) / (BB_upper − BB_lower)   ← position within bands
```

**Why**:
- `BB_width` encodes current volatility regime (wide bands = high volatility)
- `BB_pct` encodes where price sits relative to its recent range
  - Near 0 → price at lower band → potential support / oversold
  - Near 1 → price at upper band → potential resistance / overbought
  - Near 0.5 → price at mid-band

### 3.6 Returns — Return_1d, Return_5d, Return_20d

```
Return_nd[t] = (Close[t] − Close[t−n]) / Close[t−n]
```

**Why**: Raw price is non-stationary; returns are approximately stationary. Including
returns at multiple horizons lets the model capture short-term (1d), weekly (5d), and
monthly (20d) momentum simultaneously.

### 3.7 Volatility_20d

```
Volatility_20d = std(Return_1d, 20 days) × sqrt(252)
```

Annualised rolling volatility. **Why**: Volatility clustering is one of the most
well-documented stylised facts in finance. High volatility today predicts high volatility
tomorrow (GARCH effect). Including this lets LSTM and XGBoost implicitly model
heteroskedasticity without a separate volatility model.

### 3.8 Volume_ratio

```
Volume_ratio = Volume[t] / MA_20(Volume)
```

**Why**: Absolute volume is meaningless across time (average daily volume changes as
a company grows). Normalising by its own 20-day average gives a ratio: > 1 means
unusually high activity today, which often precedes large price moves.

### 3.9 Lag Features — Close_lag_1, lag_2, lag_3, lag_5, lag_10

```
Close_lag_k[t] = Close[t − k]
```

**Why**: For XGBoost (which has no built-in memory), lag features are how you provide
temporal context. The model can learn "if yesterday's price was high and today's RSI
is falling, predict a decline." LSTM doesn't need these — it builds temporal memory
internally through its gated recurrent structure.

---

## 4 · Why These Three Models?

The three models represent three fundamentally different philosophies, providing a
natural benchmark ladder.

### 4.1 ARIMA — Autoregressive Integrated Moving Average

**Philosophy**: The future is a linear function of the past values and past errors only.

```
ARIMA(p, d, q):
  d differencing steps to achieve stationarity
  p autoregressive terms: y[t] depends on y[t-1], ..., y[t-p]
  q moving average terms: y[t] depends on error[t-1], ..., error[t-q]
```

**Strengths**: Interpretable, fast, principled — has a strong statistical foundation.
**Weaknesses**: Assumes linearity. Cannot use external features. Struggles with
structural breaks and volatility clustering.
**Role in this project**: Baseline. If a complex model cannot beat ARIMA, it's not
worth its complexity.

### 4.2 XGBoost — Extreme Gradient Boosting

**Philosophy**: Ensemble of shallow decision trees, each correcting the residuals of the
previous. Can model complex non-linear interactions between features.

```
F(x) = sum of N trees, each trained on the residuals of F_1...F_{n-1}(x)
```

**Strengths**: Handles mixed feature types, robust to outliers, fast to train,
interpretable via feature importance, no assumption of stationarity.
**Weaknesses**: No built-in temporal memory — must encode time through lag features.
Cannot extrapolate beyond the training range of any feature.
**Role in this project**: Strong non-linear baseline with the full feature set.

### 4.3 LSTM — Long Short-Term Memory

**Philosophy**: Recurrent neural network with gated memory cells that learn which
information to remember and which to forget across sequences.

```
Forget gate:  f[t] = sigmoid(W_f · [h[t-1], x[t]] + b_f)
Input gate:   i[t] = sigmoid(W_i · [h[t-1], x[t]] + b_i)
Cell update:  C[t] = f[t] * C[t-1] + i[t] * tanh(W_C · [h[t-1], x[t]] + b_C)
Output gate:  h[t] = o[t] * tanh(C[t])
```

**Strengths**: Captures long-range temporal dependencies without manual lag engineering.
Can model non-linear dynamics. Benefits from the full multi-feature input matrix.
**Weaknesses**: Slow to train, sensitive to hyperparameters, requires scaling, less
interpretable than XGBoost, needs a warm-up window (seq_len = 60 days).
**Role in this project**: The most expressive model — expected to capture patterns
invisible to ARIMA and XGBoost, but must justify its computational cost.

---

## 5 · Why These Metrics?

No single metric tells the full story. Each one answers a different question.

### 5.1 MAE — Mean Absolute Error

```
MAE = (1/n) × Σ |y_true − y_pred|
```

**Answers**: On average, how many dollars off is my prediction?
**Properties**: Same unit as the target (USD). Robust to outliers (uses absolute value,
not squared). Easy to explain to a non-technical stakeholder.
**Limitation**: Treats all errors equally regardless of direction.

### 5.2 RMSE — Root Mean Squared Error

```
RMSE = sqrt((1/n) × Σ (y_true − y_pred)²)
```

**Answers**: What is the typical prediction error, penalising large mistakes heavily?
**Properties**: Same unit as target. Squaring amplifies large errors — a model with
RMSE >> MAE has occasional catastrophic predictions. Used as the primary ranking
metric in this project.
**Limitation**: Sensitive to outliers; a single extreme error dominates.

### 5.3 MAPE — Mean Absolute Percentage Error

```
MAPE = (100/n) × Σ |y_true − y_pred| / |y_true|
```

**Answers**: What percentage of the actual price is my error on average?
**Properties**: Scale-independent — allows comparing performance across different
stocks or time periods with different price levels. 2% MAPE on a $500 stock
means ~$10 typical error.
**Limitation**: Undefined when y_true = 0; asymmetric (penalises under-predictions
more than over-predictions).

### 5.4 SMAPE — Symmetric MAPE

```
SMAPE = (100/n) × Σ |y_true − y_pred| / ((|y_true| + |y_pred|) / 2)
```

**Answers**: A symmetric, bounded version of MAPE.
**Why included**: Fixes the asymmetry problem of MAPE. Bounded between 0% and 200%.

### 5.5 R² — Coefficient of Determination

```
R² = 1 − SS_res / SS_tot
     SS_res = Σ (y_true − y_pred)²
     SS_tot = Σ (y_true − mean(y_true))²
```

**Answers**: How much of the variance in the target does the model explain?
- R² = 1.0 → perfect prediction
- R² = 0.0 → model is no better than predicting the mean every time
- R² < 0.0 → model is worse than predicting the mean (bad)

**Why it matters for stocks**: A high R² on stock price (not returns) is easy to
achieve because prices are highly autocorrelated — predicting yesterday's price as
today's gets R² ≈ 0.99. Treat R² with caution; focus on RMSE and Dir.Acc.

### 5.6 Directional Accuracy (Dir.Acc)

```
Dir.Acc = fraction of days where sign(y_pred[t] - y_pred[t-1]) == sign(y_true[t] - y_true[t-1])
```

**Answers**: Does the model at least predict the right direction (up/down)?
**Why it matters**: For a trading strategy, direction matters more than magnitude.
A model with 55% directional accuracy is potentially profitable even with high RMSE.
Random guessing gives ~50%. Anything above 52–53% is practically interesting.

---

## 6 · Why Train / Val / Test Split (not Cross-Validation)?

Standard k-fold cross-validation shuffles data randomly. For time series this causes
**data leakage** — using future data to predict the past — which produces
artificially optimistic metrics.

**Solution**: A strict chronological split:
```
|←────── Train (70%) ──────→|←── Val (15%) ──→|←── Test (15%) ──→|
  used to fit the model        used to tune       held out until
                                hyperparams &      final evaluation
                                early stopping
```

- **Validation set** is used during training for early stopping (LSTM) and
  Optuna objective (XGBoost/LSTM tuning). Never seen by the model weights directly.
- **Test set** is used exactly once — final metric reporting. If you evaluate on test
  repeatedly to guide decisions, you overfit to the test set.

---

## 7 · Why MLOps? (DVC + MLflow)

### 7.1 The problem without MLOps

Without tracking:
- You retrain with different params and forget what the previous RMSE was
- You can't reproduce the exact model that gave your best result 3 weeks ago
- A colleague clones your repo but gets different data (yfinance returns live data)
- You don't know which model file corresponds to which experiment

### 7.2 What DVC solves

DVC (Data Version Control) versions **large binary files** that git cannot handle:

```
git  → versions code, configs, params.yaml, dvc.lock (small text)
DVC  → versions data/raw.csv, data/processed/*.pkl, models/*.keras (large binaries)
```

`dvc.lock` is a small text file that git tracks. It contains the MD5 hash of every
data and model file. Together, a git commit hash + dvc.lock = exact reproducibility:

```
git checkout <commit>   # restore exact code + params
dvc pull                # restore the exact data + models that commit produced
```

### 7.3 What MLflow solves

MLflow records **every training run** — params used, metrics achieved, and the model
artifact — in a queryable database (hosted on DagHub):

```
Run: XGBoost-tuning  2024-03-15 14:32
  Params: lr=0.031, max_depth=5, n_estimators=620
  Metrics: MAE=8.42, RMSE=11.3, R²=0.94
  Artifact: model/ (reloadable XGBRegressor)
```

You can compare 50 Optuna trials side by side in the DagHub UI, reload any model
by run ID, and reproduce any result exactly.

### 7.4 The two-layer system

```
DVC  → reproducibility of data + pipeline execution
MLflow → reproducibility of experiment outcomes + model artifacts
```

They complement each other: DVC ensures the data the model was trained on is
recoverable; MLflow ensures the model weights and training metrics are recoverable.
