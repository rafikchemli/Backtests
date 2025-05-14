"""Strategy implementing asset rotation between SPY and TLT
based on relative performance."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


class SpyTltRotationStrategy(StrategyBase):
    """Strategy that rotates between SPY and TLT based on their relative performance."""

    def run(self, data, initial_investment, **kwargs):
        """
        Run the SPY-TLT rotation strategy backtest.

        Parameters:
        - data: Price data for SPY and TLT
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters

        Returns:
        - Dictionary with backtest results
        """
        # Extract SPY and TLT data
        if isinstance(data, dict):
            spy_data = data["SPY"] if "SPY" in data else None
            tlt_data = data["TLT"] if "TLT" in data else None
        else:
            # Single dataframe might have been passed
            spy_data = tlt_data = data

        # Ensure we have data for both assets
        if spy_data is None or tlt_data is None:
            raise ValueError("SPY and TLT data required for this strategy")

        # Track performance
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [spy_data.index[0]]  # Use SPY dates for tracking

        # Initialize position
        current_holding = None  # Start with no position
        position_size = 0

        # For each month, compare performance and rotate
        last_month = spy_data.index[0].month
        last_year = spy_data.index[0].year

        for i in range(1, len(spy_data)):
            current_date = spy_data.index[i]
            current_month = current_date.month
            current_year = current_date.year

            # Check if we need to rotate (beginning of new month)
            if current_month != last_month or current_year != last_year:
                # Calculate returns over the last month for both assets
                spy_return = spy_data["Close"].pct_change(20).iloc[i]
                tlt_return = tlt_data["Close"].pct_change(20).iloc[i]

                # Decide which asset to hold
                if spy_return > tlt_return:
                    new_holding = "SPY"
                    entry_price = spy_data["Close"].iloc[i]
                else:
                    new_holding = "TLT"
                    entry_price = tlt_data["Close"].iloc[i]

                # If we're switching assets, record the trade
                if new_holding != current_holding:
                    # First, close existing position if any
                    if current_holding:
                        exit_price = (
                            spy_data["Close"].iloc[i - 1]
                            if current_holding == "SPY"
                            else tlt_data["Close"].iloc[i - 1]
                        )
                        portfolio = position_size * exit_price
                        trades.append(
                            (current_holding, exit_price, current_date, "SELL")
                        )

                    # Then, open new position
                    position_size = portfolio / entry_price
                    current_holding = new_holding
                    trades.append((current_holding, entry_price, current_date, "BUY"))

                # Update tracking variables
                last_month = current_month
                last_year = current_year

            # Update portfolio value based on current holding
            if current_holding == "SPY":
                portfolio = position_size * spy_data["Close"].iloc[i]
            elif current_holding == "TLT":
                portfolio = position_size * tlt_data["Close"].iloc[i]

            # Record portfolio value
            portfolio_curve.append(portfolio)
            trade_dates.append(current_date)

        # Calculate trade returns
        trade_returns = []
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                entry = trades[i][1]
                exit = trades[i + 1][1]
                trade_returns.append((exit - entry) / entry)

        return {
            "trades": trades,
            "returns": trade_returns,
            "trade_dates": trade_dates,
            "portfolio_curve": portfolio_curve,
            "strategy_name": "SPY-TLT Rotation",
            "initial_investment": initial_investment,
            "symbol": kwargs.get("symbol", ["SPY", "TLT"]),
        }

    def summarize(self, results, data=None):
        """
        Summarize and display backtest results.

        Parameters:
        - results: Dictionary with backtest results from run()
        - data: Original price data (optional)
        """
        # Use SPY as standard benchmark
        symbol = "SPY"  # Standard benchmark

        # Get benchmark data (SPY) if available
        benchmark_data = results.get("benchmark_data", None)

        # Get standardized benchmark if available
        standard_benchmark = results.get("standard_benchmark", None)

        # If no benchmark data provided, get it on demand (fallback)
        if benchmark_data is None:
            start_date = results["trade_dates"][0]
            end_date = results["trade_dates"][-1]
            benchmark_data = download_data(symbol, start_date, end_date)

        # Structure equity curves with benchmark data
        x_dates, equity, bh_equity = structure_equity_curve(
            results["trade_dates"],
            results["portfolio_curve"],
            results["initial_investment"],
            None,  # We're using multiple symbols, so no single data source
            bh_symbol=symbol,
            benchmark_data=benchmark_data,
            standard_benchmark=standard_benchmark,
        )

        n_trades = len(results["trades"]) // 2  # Count pairs of trades (buy/sell)
        avg_gain = (
            np.mean(results["returns"]) * 100 if len(results["returns"]) > 0 else 0
        )

        # Use standardized print function with standard benchmark
        print_results(
            results["strategy_name"],
            equity,
            bh_equity,
            n_trades,
            avg_gain,
            "rotation",
            standard_benchmark,
        )

        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results["strategy_name"])


if __name__ == "__main__":
    # Use already imported download_data
    symbols = ["SPY", "TLT"]
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))

    # Download each symbol
    data = {}
    for symbol in symbols:
        data[symbol] = download_data(symbol, start_date, end_date)

    # Download benchmark data (SPY) for consistency in comparison
    benchmark_data = download_data("SPY", start_date, end_date)

    strategy = SpyTltRotationStrategy()
    results = strategy.run(data, 10000)

    # Add benchmark data to results
    results["benchmark_data"] = benchmark_data

    strategy.summarize(results)
