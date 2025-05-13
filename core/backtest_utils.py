import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# General data download utility
def download_data(symbols, start_date, end_date, interval="1d", auto_adjust=True):
    if isinstance(symbols, str):
        symbols = [symbols]
    data = yf.download(symbols, start=start_date, end=end_date, interval=interval, group_by='ticker', auto_adjust=auto_adjust)
    # If only one symbol, flatten the columns
    if len(symbols) == 1:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[1] for col in data.columns]
        return data
    return data

# General performance metrics

def calculate_cagr(equity_curve, periods_per_year=252):
    if len(equity_curve) < 2:
        return 0.0
    
    # Convert to numpy array if it's a Series or other structure
    if hasattr(equity_curve, 'values'):
        equity_curve = equity_curve.values
    elif not isinstance(equity_curve, np.ndarray):
        equity_curve = np.array(equity_curve)
        
    total_return = equity_curve[-1] / equity_curve[0]
    n_years = len(equity_curve) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (total_return ** (1 / n_years) - 1) * 100

def calculate_max_drawdown(equity_curve):
    # Convert to numpy array if it's a Series or other structure
    if hasattr(equity_curve, 'values'):
        equity_curve = equity_curve.values
    elif not isinstance(equity_curve, np.ndarray):
        equity_curve = np.array(equity_curve)
        
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min() * 100 if len(dd) > 0 else 0

def structure_equity_curve(trade_dates, portfolio_curve, initial_investment, data=None, bh_symbol=None):
    equity = np.array([v if not isinstance(v, (pd.Series, np.ndarray, list)) else np.array(v).item() for v in portfolio_curve])
    x_dates = trade_dates
    # Buy and Hold
    bh_equity = [initial_investment]
    if data is not None and bh_symbol is not None:
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        start_price = data[close_col].iloc[0]
        for d in x_dates[1:]:
            if d in data.index:
                price = data[close_col].loc[d]
            else:
                price = data[close_col].iloc[data.index.get_loc(d, method='ffill')]
            bh_equity.append(initial_investment * (price / start_price))
    else:
        bh_equity = [initial_investment] * len(equity)
    # Align lengths
    min_len = min(len(x_dates), len(equity), len(bh_equity))
    x_dates = x_dates[:min_len]
    equity = equity[:min_len]
    bh_equity = bh_equity[:min_len]
    return x_dates, equity, bh_equity

def print_results(strategy_name, equity, bh_equity, n_trades, avg_gain_pct, periods_label="trade"):
    """
    Standardized function to print strategy results
    
    Parameters:
    - strategy_name: Name of the strategy
    - equity: Final equity curve 
    - bh_equity: Buy & Hold equity curve
    - n_trades: Number of trades or periods
    - avg_gain_pct: Average gain percentage per trade/period
    - periods_label: Label for the periods (e.g., "trade", "period", "month")
    """
    # Ensure we're working with arrays
    if hasattr(equity, 'values'):
        equity_arr = equity.values
    elif not isinstance(equity, np.ndarray):
        equity_arr = np.array(equity)
    else:
        equity_arr = equity
        
    if hasattr(bh_equity, 'values'):
        bh_equity_arr = bh_equity.values
    elif not isinstance(bh_equity, np.ndarray):
        bh_equity_arr = np.array(bh_equity)
    else:
        bh_equity_arr = bh_equity
    
    cagr = calculate_cagr(equity_arr)
    max_dd = calculate_max_drawdown(equity_arr)
    bh_cagr = calculate_cagr(bh_equity_arr)
    bh_max_dd = calculate_max_drawdown(bh_equity_arr)
    
    print(f"\n===== {strategy_name} =====")
    print(f"Trades: {n_trades}")
    print(f"Average gain per {periods_label}: {avg_gain_pct:.2f}%")
    print(f"CAGR (approx): {cagr:.2f}%")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Final equity: ${equity_arr[-1]:.2f}")
    print(f"Buy & Hold CAGR: {bh_cagr:.2f}%")
    print(f"Buy & Hold Max Drawdown: {bh_max_dd:.2f}%")
    print(f"Buy & Hold Final Equity: ${bh_equity_arr[-1]:.2f}") 