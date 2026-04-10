from __future__ import annotations

import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# ARIMA grid-search  (minimise AIC)
# ---------------------------------------------------------------------------

def tune_arima(
    series: pd.Series,
    p_values: list | None = None,
    d_values: list | None = None,
    q_values: list | None = None,
    verbose: bool = True,
) -> tuple[tuple, pd.DataFrame]:
    """Grid-search over ARIMA (p, d, q) and return the best order by AIC.

    Args:
        series:   Univariate time series.
        p_values: AR orders to try  (default 0-3).
        d_values: Differencing orders (default 0-2).
        q_values: MA orders to try  (default 0-3).
        verbose:  Print progress.

    Returns:
        best_order: (p, d, q) with lowest AIC.
        results_df: All attempted orders sorted by AIC.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if p_values is None:
        p_values = [0, 1, 2, 3]
    if d_values is None:
        d_values = [0, 1, 2]
    if q_values is None:
        q_values = [0, 1, 2, 3]

    records = []
    best_aic, best_order = np.inf, None

    for order in itertools.product(p_values, d_values, q_values):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(series, order=order).fit()
            records.append({"order": order, "AIC": fitted.aic, "BIC": fitted.bic})
            if fitted.aic < best_aic:
                best_aic, best_order = fitted.aic, order
            if verbose:
                print(f"ARIMA{order}  AIC={fitted.aic:.2f}")
        except Exception:
            continue

    if not records:
        raise RuntimeError(
            "All ARIMA fits failed. "
            "Check that p_values/d_values/q_values are lists of integers "
            "(e.g. p_values=[0,1,2]), not a pandas Series."
        )
    results_df = pd.DataFrame(records).sort_values("AIC").reset_index(drop=True)
    if verbose:
        print(f"\nBest order: ARIMA{best_order}  (AIC={best_aic:.2f})")
    return best_order, results_df


# ---------------------------------------------------------------------------
# XGBoost  —  Optuna
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    show_progress: bool = True,
) -> tuple[dict, object]:
    """Tune XGBRegressor hyper-parameters with Optuna.

    Returns:
        best_params: dict of best hyper-parameters.
        study:       Optuna Study object.
    """
    import optuna
    from xgboost import XGBRegressor

    if not show_progress:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":       42,
            "verbosity":          0,
            "early_stopping_rounds": 30,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best XGB RMSE: {study.best_value ** 0.5:.6f}")
    return study.best_params, study


# ---------------------------------------------------------------------------
# LSTM  —  Optuna
# ---------------------------------------------------------------------------

def tune_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
    max_epochs: int = 50,
    show_progress: bool = True,
) -> tuple[dict, object]:
    """Tune LSTM hyper-parameters with Optuna.

    Returns:
        best_params: dict of best hyper-parameters.
        study:       Optuna Study object.
    """
    import optuna
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    input_shape = X_train.shape[1:]

    if show_progress:
        print(f"LSTM Optuna tuning: {n_trials} trials × up to {max_epochs} epochs each …")

    def objective(trial):
        units_1 = trial.suggest_categorical("units_1", [32, 64, 128])
        units_2 = trial.suggest_categorical("units_2", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr      = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch   = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(units_1, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units_2),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )
        es = EarlyStopping(monitor="val_loss", patience=10,
                           restore_best_weights=True, verbose=0)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch,
            callbacks=[es],
            verbose=0,
        )
        val_loss = min(history.history["val_loss"])
        n_epochs = len(history.history["val_loss"])
        if show_progress:
            print(f"  Trial {trial.number:2d}  units=({units_1},{units_2})"
                  f"  dropout={dropout:.2f}  lr={lr:.5f}  batch={batch:2d}"
                  f"  val_loss={val_loss:.5f}  ({n_epochs} ep)")
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    print(f"Best LSTM val_loss: {study.best_value:.6f}")
    return study.best_params, study
