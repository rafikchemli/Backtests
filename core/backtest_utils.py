import importlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Data management utilities


def get_cached_data(
    symbols, start_date, end_date, data_cache, interval="1d", auto_adjust=True
):
    """
    Get cached data or download if not available.

    Parameters:
    - symbols: Symbol or list of symbols to get data for
    - start_date: Start date for the data
    - end_date: End date for the data
    - data_cache: Dictionary to use for caching
    - interval: Data interval (default: "1d")
    - auto_adjust: Whether to auto-adjust data (default: True)

    Returns:
    - Data for the requested symbol(s)
    """
    if isinstance(symbols, str):
        # Single symbol
        key = f"{symbols}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if key not in data_cache:
            data_cache[key] = download_data(
                symbols, start_date, end_date, interval, auto_adjust
            )
        return data_cache[key]
    else:
        # List of symbols
        all_cached = True
        symbols_data = {}

        # Check if all symbols are cached
        for s in symbols:
            key = f"{s}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            if key not in data_cache:
                all_cached = False
                break
            symbols_data[s] = data_cache[key]

        if all_cached:
            return symbols_data

        # Download all symbols at once
        combined_data = download_data(
            symbols, start_date, end_date, interval, auto_adjust
        )

        # Cache individual symbols and prepare return data
        result_data = {}
        for s in symbols:
            key = f"{s}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            if isinstance(combined_data, pd.DataFrame) and isinstance(
                combined_data.columns, pd.MultiIndex
            ):
                data_cache[key] = combined_data[s].dropna()
                result_data[s] = data_cache[key]
            else:
                data_cache[key] = combined_data
                result_data = combined_data

        return result_data


def calculate_date_range(years_back):
    """
    Calculate standard date range for backtesting.

    Parameters:
    - years_back: Number of years to go back

    Returns:
    - Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    # Add 1 extra year to ensure we have enough data for calculations
    start_date = end_date - timedelta(days=365 * (years_back + 1))
    return start_date, end_date


def load_strategy(strategy_key, strategies_config):
    """
    Load a strategy class based on configuration.

    Parameters:
    - strategy_key: Key identifying the strategy in the config dictionary
    - strategies_config: Dictionary mapping strategy keys to their configurations

    Returns:
    - Tuple of (strategy_class, symbol, years_back, initial_investment)
    """
    if strategy_key not in strategies_config:
        raise ValueError(f"Strategy '{strategy_key}' not found in configuration")

    module_path, class_name, symbol, years_back, initial_investment = strategies_config[
        strategy_key
    ]

    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        return strategy_class, symbol, years_back, initial_investment
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load strategy '{strategy_key}': {str(e)}")


# General data download utility
def download_data(symbols, start_date, end_date, interval="1d", auto_adjust=True):
    if isinstance(symbols, str):
        symbols = [symbols]
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker",
        auto_adjust=auto_adjust,
    )
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
    if hasattr(equity_curve, "values"):
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
    if hasattr(equity_curve, "values"):
        equity_curve = equity_curve.values
    elif not isinstance(equity_curve, np.ndarray):
        equity_curve = np.array(equity_curve)

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min() * 100 if len(dd) > 0 else 0


def structure_equity_curve(
    trade_dates,
    portfolio_curve,
    initial_investment,
    data=None,
    bh_symbol=None,
    benchmark_data=None,
    standard_benchmark=None,
):
    """
    Structure the equity curve and calculate buy & hold benchmark performance.

    Parameters:
    - trade_dates: List of dates for the strategy
    - portfolio_curve: List of portfolio values
    - initial_investment: Initial investment amount
    - data: Data used for the strategy
    - bh_symbol: Symbol for buy & hold calculation
    - benchmark_data: Optional benchmark data to use for consistent comparison
    - standard_benchmark: Optional dictionary with standard benchmark values

    Returns:
    - Tuple of (x_dates, equity, bh_equity)
    """
    # Convert strategy equity to array
    equity = np.array(
        [
            (
                v
                if not isinstance(v, (pd.Series, np.ndarray, list))
                else np.array(v).item()
            )
            for v in portfolio_curve
        ]
    )
    x_dates = trade_dates

    # Buy and Hold equity calculation
    if standard_benchmark is not None:
        # Use standard benchmark for consistent buy & hold across strategies
        bh_equity = []
        for date in x_dates:
            # Find closest date in standard benchmark
            if date in standard_benchmark:
                bh_equity.append(standard_benchmark[date])
            else:
                # Try to find closest date
                closest_dates = sorted(
                    standard_benchmark.keys(),
                    key=lambda d: abs((d - date).total_seconds()),
                )
                if closest_dates:
                    bh_equity.append(standard_benchmark[closest_dates[0]])
                else:
                    # Use initial investment if no close match
                    bh_equity.append(initial_investment)

    # Use provided benchmark data if no standard benchmark
    elif benchmark_data is not None and bh_symbol is not None:
        close_col = (
            "Close" if "Close" in benchmark_data.columns else benchmark_data.columns[0]
        )
        start_price = None
        bh_equity = [initial_investment]

        for d in x_dates:
            if d in benchmark_data.index:
                if start_price is None:
                    start_price = benchmark_data.loc[d, close_col]
                price = benchmark_data.loc[d, close_col]
                bh_equity.append(initial_investment * (price / start_price))
            else:
                # Try to find closest date
                idx = benchmark_data.index.get_indexer([d], method="ffill")[0]
                if idx >= 0:
                    if start_price is None:
                        start_price = benchmark_data.iloc[idx][close_col]
                    price = benchmark_data.iloc[idx][close_col]
                    bh_equity.append(initial_investment * (price / start_price))
                else:
                    # If no close match, use last value
                    bh_equity.append(bh_equity[-1] if bh_equity else initial_investment)

        # Remove the first element if we have enough values
        if len(bh_equity) > len(x_dates):
            bh_equity = bh_equity[1:]

    # Use provided strategy data if no benchmark data
    elif data is not None and bh_symbol is not None:
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        start_price = data[close_col].iloc[0]
        bh_equity = [initial_investment]
        for d in x_dates[1:]:
            if d in data.index:
                price = data[close_col].loc[d]
            else:
                price = data[close_col].iloc[data.index.get_loc(d, method="ffill")]
            bh_equity.append(initial_investment * (price / start_price))
    else:
        bh_equity = [initial_investment] * len(equity)

    # Align lengths
    min_len = min(len(x_dates), len(equity), len(bh_equity))
    x_dates = x_dates[:min_len]
    equity = equity[:min_len]
    bh_equity = bh_equity[:min_len]

    return x_dates, equity, bh_equity


def print_results(
    strategy_name,
    equity,
    bh_equity,
    n_trades,
    avg_gain_pct,
    periods_label="trade",
    standard_benchmark=None,
):
    """
    Standardized function to print strategy results

    Parameters:
    - strategy_name: Name of the strategy
    - equity: Final equity curve
    - bh_equity: Buy & Hold equity curve
    - n_trades: Number of trades or periods
    - avg_gain_pct: Average gain percentage per trade/period
    - periods_label: Label for the periods (e.g., "trade", "period", "month")
    - standard_benchmark: Optional standard benchmark to use for final equity value
    """
    # Ensure we're working with arrays
    if hasattr(equity, "values"):
        equity_arr = equity.values
    elif not isinstance(equity, np.ndarray):
        equity_arr = np.array(equity)
    else:
        equity_arr = equity

    if hasattr(bh_equity, "values"):
        bh_equity_arr = bh_equity.values
    elif not isinstance(bh_equity, np.ndarray):
        bh_equity_arr = np.array(bh_equity)
    else:
        bh_equity_arr = bh_equity

    cagr = calculate_cagr(equity_arr)
    max_dd = calculate_max_drawdown(equity_arr)
    bh_cagr = calculate_cagr(bh_equity_arr)
    bh_max_dd = calculate_max_drawdown(bh_equity_arr)

    # Get the standardized final benchmark value if available
    bh_final_equity = bh_equity_arr[-1]
    if standard_benchmark is not None:
        # Use the final value from the standard benchmark for consistent reporting
        last_date = max(standard_benchmark.keys())
        bh_final_equity = standard_benchmark[last_date]

    print(f"\n===== {strategy_name} =====")
    print(f"Trades: {n_trades}")
    print(f"Average gain per {periods_label}: {avg_gain_pct:.2f}%")
    print(f"CAGR (approx): {cagr:.2f}%")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Final equity: ${equity_arr[-1]:.2f}")
    print(f"Buy & Hold CAGR: {bh_cagr:.2f}%")
    print(f"Buy & Hold Max Drawdown: {bh_max_dd:.2f}%")
    print(f"Buy & Hold Final Equity: ${bh_final_equity:.2f}")
