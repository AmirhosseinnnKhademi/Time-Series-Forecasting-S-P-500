import pandas as pd
import yfinance as yf
from datetime import datetime


def get_stock_data(delta: int, tickers: list) -> pd.DataFrame:
    """Download historical OHLCV data for given tickers.

    Args:
        delta:   Number of years of history to fetch.
        tickers: List of ticker symbols, e.g. ['^GSPC', 'AAPL'].

    Returns:
        Long-format DataFrame with a 'ticker' column added.
    """
    end = datetime.now()
    start = end.replace(year=end.year - delta)

    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        # yfinance >=0.2.50 returns MultiIndex columns even for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.assign(ticker=ticker)
        frames.append(df)

    return pd.concat(frames).sort_index()


def get_sp500_data(delta: int = 5) -> pd.DataFrame:
    """Convenience wrapper — downloads S&P 500 index (^GSPC)."""
    return get_stock_data(delta, ['^GSPC'])


def get_tech_stocks(delta: int = 5) -> pd.DataFrame:
    """Download a basket of large-cap tech stocks."""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    return get_stock_data(delta, tickers)
