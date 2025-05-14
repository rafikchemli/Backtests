"""Strategy implementing the 'Sell in May and Go Away'
trading logic on financial data."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


def is_month_in_range(date, start_month, end_month):
    """
    Check if a date's month falls within a specific month range.

    Parameters:
    - date: The date to check
    - start_month: Starting month (1-12)
    - end_month: Ending month (1-12)

    Returns:
    - Boolean indicating if the date's month is in range
    """
    month = date.month
    if start_month <= end_month:
        return start_month <= month <= end_month
    else:
        # Handle range that crosses year boundary (e.g., Nov-Feb)
        return month >= start_month or month <= end_month


class SellMayStrategy(StrategyBase):
    """
    Strategy implementing the 'Sell in May and Go Away' effect.

    Invests in stocks from November through April, then moves to cash or bonds
    from May through October.
    """

    def run(self, data, initial_investment, **kwargs):
        """
        Run the Sell in May strategy backtest.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters (e.g., symbol)

        Returns:
        - Dictionary with backtest results
        """
        if isinstance(data, dict):
            data = data["SPY"] if "SPY" in data else next(iter(data.values()))

        close_col = "Close" if "Close" in data.columns else data.columns[0]
        data = data.sort_index()

        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]

        # Define favorable and unfavorable periods
        favorable_start_month = 11  # November
        favorable_end_month = 4  # April

        # Track state
        in_market = False
        entry_price = None
        # entry_date = None

        # Process data day by day
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data[close_col].iloc[i]

            # Check if this is a favorable period
            is_favorable = is_month_in_range(
                current_date, favorable_start_month, favorable_end_month
            )

            # Determine actions based on current state
            if is_favorable and not in_market:
                # Enter the market at the start of favorable period
                entry_price = current_price
                # entry_date = current_date
                position_size = portfolio / entry_price
                in_market = True
                trades.append((entry_price, None, current_date, "BUY"))

            elif not is_favorable and in_market:
                # Exit the market at the end of favorable period
                exit_price = current_price
                portfolio = position_size * exit_price
                position_size = 0
                in_market = False

                # Update the last buy with this exit price
                for j in range(len(trades) - 1, -1, -1):
                    if trades[j][3] == "BUY" and trades[j][1] is None:
                        trades[j] = (
                            trades[j][0],
                            exit_price,
                            trades[j][2],
                            "BUY-SELL",
                        )
                        break

            # Update portfolio value based on market position
            if in_market:
                portfolio = position_size * current_price

            # Track portfolio value
            portfolio_curve.append(portfolio)
            trade_dates.append(current_date)

        # Close position at the end if still in market
        if in_market:
            exit_price = data[close_col].iloc[-1]
            portfolio = position_size * exit_price
            portfolio_curve[-1] = portfolio

            for j in range(len(trades) - 1, -1, -1):
                if trades[j][3] == "BUY" and trades[j][1] is None:
                    trades[j] = (trades[j][0], exit_price, trades[j][2], "BUY-SELL")
                    break

        # Calculate returns for completed trades
        trade_returns = []
        for entry, exit, _, trade_type in trades:
            if entry is not None and exit is not None:
                trade_returns.append((exit - entry) / entry)

        return {
            "trades": trades,
            "returns": trade_returns,
            "trade_dates": trade_dates,
            "portfolio_curve": portfolio_curve,
            "strategy_name": "Sell in May",
            "initial_investment": initial_investment,
            "symbol": kwargs.get("symbol", "SPY"),
        }

    def summarize(self, results, data=None):
        """
        Summarize and display backtest results.

        Parameters:
        - results: Dictionary with backtest results from run()
        - data: Original price data (optional)
        """
        # Use SPY as standard benchmark
        symbol = "SPY"

        # Get benchmark data (SPY) if available
        benchmark_data = results.get("benchmark_data", None)

        # Get standardized benchmark if available
        standard_benchmark = results.get("standard_benchmark", None)

        # If no benchmark data provided, get it on demand (fallback)
        if benchmark_data is None:
            start_date = results["trade_dates"][0]
            end_date = results["trade_dates"][-1]
            benchmark_data = download_data(symbol, start_date, end_date)

        # Structure equity curves
        x_dates, equity, bh_equity = structure_equity_curve(
            results["trade_dates"],
            results["portfolio_curve"],
            results["initial_investment"],
            data,
            bh_symbol=symbol,
            benchmark_data=benchmark_data,
            standard_benchmark=standard_benchmark,
        )

        # Calculate metrics
        n_trades = len([t for t in results["trades"] if t[3] in ["BUY", "BUY-SELL"]])
        avg_gain = (
            np.mean(results["returns"]) * 100 if len(results["returns"]) > 0 else 0
        )

        # Use standardized print function
        print_results(
            results["strategy_name"],
            equity,
            bh_equity,
            n_trades,
            avg_gain,
            "season",
            standard_benchmark,
        )

        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results["strategy_name"])


if __name__ == "__main__":
    # Use already imported download_data
    symbol = "SPY"
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)

    # Download benchmark data for consistency
    benchmark_data = download_data("SPY", start_date, end_date)

    strategy = SellMayStrategy()
    results = strategy.run(data, 10000)

    # Add benchmark data to results
    results["benchmark_data"] = benchmark_data

    strategy.summarize(results, data)
