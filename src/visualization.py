from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(path: str | None) -> None:
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Forecast plots
# ---------------------------------------------------------------------------

def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates=None,
    title: str = "Forecast vs Actual",
    save_path: str | None = None,
) -> None:
    """Actual vs predicted line chart with residual panel."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    x = dates if dates is not None else np.arange(len(y_true))

    axes[0].plot(x, y_true, label="Actual",    color="steelblue", linewidth=1.5)
    axes[0].plot(x, y_pred, label="Predicted", color="tomato",    linewidth=1.5, alpha=0.85)
    axes[0].fill_between(x, y_true, y_pred, alpha=0.15, color="orange")
    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    residuals = y_true - y_pred
    axes[1].bar(x, residuals, color=np.where(residuals >= 0, "steelblue", "tomato"),
                alpha=0.7, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.3)

    if dates is not None:
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    plt.tight_layout()
    _save_or_show(save_path)


def plot_model_comparison(
    results: list[dict],
    metrics: list | None = None,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing multiple models across key metrics."""
    if metrics is None:
        metrics = ["MAE", "RMSE", "MAPE", "R²"]

    df = pd.DataFrame(results).set_index("Model")
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", len(df))

    for ax, metric in zip(axes, available):
        bars = ax.bar(df.index, df[metric], color=palette, edgecolor="white")
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history,
    save_path: str | None = None,
) -> None:
    """Plot LSTM training / validation loss and MAE curves."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    has_mae = "mae" in hist
    n_cols = 2 if has_mae else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(epochs, hist["loss"],     label="Train Loss", color="steelblue")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss",   color="tomato")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if has_mae:
        axes[1].plot(epochs, hist["mae"],     label="Train MAE", color="steelblue")
        axes[1].plot(epochs, hist["val_mae"], label="Val MAE",   color="tomato")
        axes[1].set_title("MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.suptitle("LSTM Training Curves", fontsize=13)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n // 2)))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_title(f"Top-{top_n} Feature Importances (XGBoost)")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: str | None = None,
) -> None:
    """Four-panel residual diagnostic plot."""
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Residuals over time
    axes[0, 0].plot(residuals, color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].grid(alpha=0.3)

    # Histogram
    sns.histplot(residuals, bins=50, kde=True, ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].grid(alpha=0.3)

    # Predicted vs Actual
    axes[1, 0].scatter(y_pred, y_true, alpha=0.4, s=10, color="steelblue")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1, 0].plot([mn, mx], [mn, mx], "r--", linewidth=1)
    axes[1, 0].set_title("Predicted vs Actual")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].grid(alpha=0.3)

    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.4, s=10, color="steelblue")
    axes[1, 1].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Residuals vs Predicted")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Interactive candlestick  (Plotly)
# ---------------------------------------------------------------------------

def plot_candlestick(
    df: pd.DataFrame,
    ticker: str = "^GSPC",
    last_n_days: int = 252,
    save_path: str | None = None,
):
    """Interactive Plotly candlestick with volume sub-chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly is not installed. Run: pip install plotly")
        return

    df_plot = df.tail(last_n_days).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"],   close=df_plot["Close"],
            name=ticker,
        ),
        row=1, col=1,
    )

    if "Volume" in df_plot.columns:
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(df_plot["Close"], df_plot["Open"])
        ]
        fig.add_trace(
            go.Bar(x=df_plot.index, y=df_plot["Volume"],
                   marker_color=colors, name="Volume", opacity=0.6),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{ticker} — Last {last_n_days} trading days",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def plot_forecast_with_history(
    full_series,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    train_end_date,
    val_end_date,
    dates_test=None,
    title: str = "Full History + Forecast",
    save_path: str | None = None,
) -> None:
    """Full price history with train/val/test regions and forecast overlay.

    Draws shaded bands for each split, vertical dashed boundary lines,
    and overlays the predicted vs actual on the test window.

    Args:
        full_series:    pd.Series (or array) of the entire Close price history.
        y_true_test:    Actual test values (original scale).
        y_pred_test:    Model predictions on test set (original scale).
        train_end_date: Last date of the training set (used for dividing line).
        val_end_date:   Last date of the validation set.
        dates_test:     DatetimeIndex for the test window (optional).
        title:          Chart title.
        save_path:      If provided, save to this path instead of showing.
    """
    y_true_test = np.asarray(y_true_test).ravel()
    y_pred_test = np.asarray(y_pred_test).ravel()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # ── full history (background) ──────────────────────────────────────────
    if hasattr(full_series, "index"):
        x_full = full_series.index
        y_full = full_series.values
    else:
        x_full = np.arange(len(full_series))
        y_full = np.asarray(full_series)

    axes[0].plot(x_full, y_full, color="lightgrey", linewidth=1.2,
                 label="Full history", zorder=1)

    # ── shaded split regions ──────────────────────────────────────────────
    axes[0].axvspan(x_full[0], train_end_date,
                    alpha=0.08, color="steelblue", label="Train")
    axes[0].axvspan(train_end_date, val_end_date,
                    alpha=0.12, color="orange", label="Validation")
    axes[0].axvspan(val_end_date, x_full[-1],
                    alpha=0.10, color="green", label="Test")

    # ── split boundary lines ──────────────────────────────────────────────
    for date, label, color in [
        (train_end_date, "Train end", "steelblue"),
        (val_end_date,   "Val end",   "darkorange"),
    ]:
        axes[0].axvline(date, color=color, linestyle="--", linewidth=1.4, alpha=0.8)
        axes[0].text(date, axes[0].get_ylim()[1] if axes[0].get_ylim()[1] != 1.0 else y_full.max(),
                     f"  {label}", color=color, fontsize=8, va="top", rotation=90)

    # ── forecast overlay on test window ──────────────────────────────────
    x_test = dates_test if dates_test is not None else np.arange(len(y_true_test))
    axes[0].plot(x_test, y_true_test, color="steelblue", linewidth=1.8,
                 label="Actual (test)", zorder=3)
    axes[0].plot(x_test, y_pred_test, color="tomato", linewidth=1.8,
                 linestyle="--", label="Predicted (test)", zorder=3)
    axes[0].fill_between(x_test, y_true_test, y_pred_test,
                         alpha=0.18, color="orange", zorder=2)

    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── residual bar (test only) ──────────────────────────────────────────
    residuals = y_true_test - y_pred_test
    axes[1].bar(x_test, residuals,
                color=np.where(residuals >= 0, "steelblue", "tomato"),
                alpha=0.7, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    if dates_test is not None:
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    plt.tight_layout()
    _save_or_show(save_path)


def plot_forecast_interactive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates=None,
    title: str = "Forecast vs Actual",
    save_path: str | None = None,
):
    """Interactive Plotly forecast chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed. Run: pip install plotly")
        return

    x = list(dates) if dates is not None else list(range(len(y_true)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, name="Actual",
                             line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_pred, name="Predicted",
                             line=dict(color="tomato", width=2, dash="dot")))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", height=450,
    )

    if save_path:
        fig.write_html(save_path)
    return fig
