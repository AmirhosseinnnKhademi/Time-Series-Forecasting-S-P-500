from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ---------------------------------------------------------------------------
# Tabular summaries
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for all numeric columns."""
    return df.describe().T


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns that have at least one missing value."""
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    report = pd.DataFrame({"count": missing, "pct": pct}).query("count > 0")
    return report.sort_values("pct", ascending=False)


def returns_summary(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Annualised return, volatility, Sharpe, max-drawdown, skew, kurtosis."""
    r = df[close_col].pct_change().dropna()
    annual_ret  = r.mean() * 252
    annual_vol  = r.std() * np.sqrt(252)
    sharpe      = annual_ret / annual_vol if annual_vol else np.nan
    cum         = (1 + r).cumprod()
    roll_max    = cum.cummax()
    drawdown    = (cum - roll_max) / roll_max
    max_dd      = drawdown.min()
    return pd.DataFrame({
        "Annual Return": [annual_ret],
        "Annual Volatility": [annual_vol],
        "Sharpe Ratio": [sharpe],
        "Max Drawdown": [max_dd],
        "Skewness": [r.skew()],
        "Kurtosis": [r.kurtosis()],
    })


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, verbose: bool = True) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Lags used": result[2],
        "Observations": result[3],
        "Stationary (5%)": result[1] < 0.05,
    }
    out.update({f"Critical {k}": v for k, v in result[4].items()})
    if verbose:
        print(pd.Series(out).to_string())
    return out


def kpss_test(series: pd.Series, verbose: bool = True) -> dict:
    """KPSS test — null hypothesis: series IS stationary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
    out = {
        "KPSS Statistic": stat,
        "p-value": p,
        "Lags used": lags,
        "Stationary (5%)": p > 0.05,
    }
    out.update({f"Critical {k}": v for k, v in crit.items()})
    if verbose:
        print(pd.Series(out).to_string())
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_or_show(path: str | None) -> None:
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_price_history(
    df: pd.DataFrame,
    close_col: str = "Close",
    ticker: str = "",
    ma_windows: list | None = None,
    save_path: str | None = None,
) -> None:
    """Closing price with optional moving-average overlays."""
    if ma_windows is None:
        ma_windows = [50, 200]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    ax1.plot(df.index, df[close_col], label="Close", linewidth=1.2, color="steelblue")
    for w in ma_windows:
        col = f"MA_{w}"
        if col in df.columns:
            ax1.plot(df.index, df[col], label=f"MA {w}", linewidth=1, alpha=0.8)
    ax1.set_title(f"{ticker} Price History" if ticker else "Price History", fontsize=14)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    if "Volume" in df.columns:
        ax2.bar(df.index, df["Volume"], color="grey", alpha=0.5, width=1)
        ax2.set_ylabel("Volume")
        ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_returns_distribution(
    df: pd.DataFrame,
    close_col: str = "Close",
    save_path: str | None = None,
) -> None:
    """Histogram + KDE of daily returns with normal overlay."""
    returns = df[close_col].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE
    sns.histplot(returns, bins=80, kde=True, stat="density",
                 ax=axes[0], color="steelblue", alpha=0.6)
    x = np.linspace(returns.min(), returns.max(), 200)
    axes[0].plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 "r--", label="Normal fit")
    axes[0].set_title("Daily Returns Distribution")
    axes[0].set_xlabel("Return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Q-Q plot
    from scipy import stats as sp_stats
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(returns, dist="norm")
    axes[1].scatter(osm, osr, alpha=0.4, s=10, color="steelblue")
    axes[1].plot(osm, slope * np.array(osm) + intercept, "r--")
    axes[1].set_title("Q-Q Plot (daily returns)")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list | None = None,
    save_path: str | None = None,
) -> None:
    """Correlation heat-map for the selected feature columns."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(10, len(cols) // 2), max(8, len(cols) // 2)))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlGn",
                center=0, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_decomposition(
    series: pd.Series,
    period: int = 252,
    model: str = "additive",
    save_path: str | None = None,
) -> None:
    """Trend / seasonal / residual decomposition."""
    dec = seasonal_decompose(series.dropna(), model=model, period=period)
    fig = dec.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle("Time Series Decomposition", y=1.01, fontsize=13)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 50,
    save_path: str | None = None,
) -> None:
    """Side-by-side ACF and PACF plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05)
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_rolling_stats(
    series: pd.Series,
    windows: list | None = None,
    save_path: str | None = None,
) -> None:
    """Rolling mean and rolling std for multiple windows."""
    if windows is None:
        windows = [30, 90, 252]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(series.index, series, label="Original", alpha=0.6, linewidth=0.8)
    for w in windows:
        axes[0].plot(series.index, series.rolling(w).mean(), label=f"Mean {w}d", linewidth=1.2)
    axes[0].set_title("Rolling Mean")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for w in windows:
        axes[1].plot(series.index, series.rolling(w).std(), label=f"Std {w}d", linewidth=1.2)
    axes[1].set_title("Rolling Std Dev")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)
