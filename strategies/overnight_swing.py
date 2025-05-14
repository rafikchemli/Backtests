"""Strategy implementing the Overnight Swing trading logic on financial data."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


class OvernightSwingStrategy(StrategyBase):
    """Strategy that buys at market close and sells at the next day's open."""

    def run(self, data, initial_investment, **kwargs):
        """
        Run the Overnight Swing strategy backtest.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters (e.g., symbol)

        Returns:
        - Dictionary with backtest results
        """
        trades, returns, trade_dates, portfolio_curve = self.backtest_overnight(
            data, initial_investment
        )
        return {
            "trades": trades,
            "returns": returns,
            "trade_dates": trade_dates,
            "portfolio_curve": portfolio_curve,
            "strategy_name": "Overnight Swing",
            "initial_investment": initial_investment,
            "symbol": kwargs.get("symbol", "SPY"),
        }

    def backtest_overnight(self, data, initial_investment):
        """
        Implement Overnight Swing strategy logic.

        Buy at close and sell at next day's open.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest

        Returns:
        - Tuple of (trades, returns, trade_dates, portfolio_curve)
        """
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]  # Start with the first date in the data
        for i in range(1, len(data)):
            entry = data["Close"].iloc[i - 1]
            exit = data["Open"].iloc[i]
            trade_return = (exit - entry) / entry
            portfolio = portfolio * (1 + trade_return)
            trades.append((entry, exit, data.index[i]))
            portfolio_curve.append(portfolio)
            trade_dates.append(data.index[i])
        returns = [(exit - entry) / entry for entry, exit, _ in trades]
        return trades, returns, trade_dates, portfolio_curve

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
            data,
            bh_symbol=symbol,
            benchmark_data=benchmark_data,
        )

        n_trades = len(results["returns"])
        avg_gain = np.mean(results["returns"]) * 100 if n_trades > 0 else 0

        # Use standardized print function with standard benchmark
        print_results(
            results["strategy_name"],
            equity,
            bh_equity,
            n_trades,
            avg_gain,
            "trade",
            standard_benchmark,
        )

        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results["strategy_name"])


if __name__ == "__main__":
    # Use already imported download_data
    symbol = "SPY"
    years_back = 20  # Match main.py setting
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)

    # Download benchmark data for consistency
    benchmark_data = download_data("SPY", start_date, end_date)

    strategy = OvernightSwingStrategy()
    results = strategy.run(data, 10000)

    # Add benchmark data to results
    results["benchmark_data"] = benchmark_data

    strategy.summarize(results, data)
