from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def train_arima(series: pd.Series, order: tuple = (1, 1, 1)):
    """Fit an ARIMA model.

    Args:
        series: Univariate time series (Close prices).
        order:  (p, d, q) tuple.

    Returns:
        Fitted ARIMAResults object.
    """
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=order)
    return model.fit()


def arima_rolling_forecast(
    series: pd.Series,
    train_size: int,
    order: tuple = (1, 1, 1),
) -> np.ndarray:
    """Walk-forward one-step-ahead forecasts on the test window.

    Args:
        series:     Full series (train + test).
        train_size: Number of initial training observations.
        order:      ARIMA (p, d, q).

    Returns:
        Array of one-step forecasts aligned with series[train_size:].
    """
    from statsmodels.tsa.arima.model import ARIMA

    history = list(series.iloc[:train_size])
    forecasts = []

    for t in range(train_size, len(series)):
        model  = ARIMA(history, order=order)
        fitted = model.fit()
        yhat   = fitted.forecast(steps=1)[0]
        forecasts.append(yhat)
        history.append(series.iloc[t])

    return np.array(forecasts)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
):
    """Train an XGBRegressor with early stopping.

    Args:
        X_train, y_train: Training features and targets.
        X_val,   y_val:   Validation features and targets (for early stopping).
        params:           XGBRegressor kwargs; sensible defaults used if None.

    Returns:
        Fitted XGBRegressor.
    """
    from xgboost import XGBRegressor

    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )
    if params:
        defaults.update(params)

    model = XGBRegressor(**defaults)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def build_lstm(
    input_shape: tuple,
    units: tuple = (64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """Build a stacked LSTM model.

    Args:
        input_shape:    (seq_len, n_features).
        units:          Number of units in each LSTM layer.
        dropout:        Dropout rate after each LSTM layer.
        learning_rate:  Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    layer_list = [
        layers.Input(shape=input_shape),
        layers.LSTM(units[0], return_sequences=len(units) > 1),
        layers.Dropout(dropout),
    ]
    for i, u in enumerate(units[1:]):
        layer_list.append(layers.LSTM(u, return_sequences=(i < len(units) - 2)))
        layer_list.append(layers.Dropout(dropout))
    layer_list += [layers.Dense(16, activation="relu"), layers.Dense(1)]
    model = keras.Sequential(layer_list)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
):
    """Train the LSTM with early stopping.

    Returns:
        (trained_model, history)
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history
