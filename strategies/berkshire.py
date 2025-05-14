"""Strategy for backtesting Berkshire Hathaway against SPY."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


class BerkshireStrategy(StrategyBase):
    """Strategy that simulates investing in Berkshire Hathaway
    with a buy and hold approach."""

    def run(self, data, initial_investment, **kwargs):
        """
        Run the Berkshire Hathaway strategy backtest.

        Parameters:
        - data: Price data for Berkshire Hathaway
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters

        Returns:
        - Dictionary with backtest results
        """
        close_col = "Close" if "Close" in data.columns else data.columns[0]

        # Buy and hold strategy - simply track portfolio value over time
        portfolio_curve = []
        trade_dates = []

        start_price = data[close_col].iloc[0]
        for i, date in enumerate(data.index):
            price = data[close_col].iloc[i]
            portfolio = initial_investment * (price / start_price)
            portfolio_curve.append(portfolio)
            trade_dates.append(date)

        # No actual trades, just buy and hold
        trades = [(start_price, data[close_col].iloc[-1], trade_dates[0], "BUY-HOLD")]

        return {
            "trades": trades,
            "returns": [(data[close_col].iloc[-1] - start_price) / start_price],
            "trade_dates": trade_dates,
            "portfolio_curve": portfolio_curve,
            "strategy_name": "Berkshire Hathaway",
            "initial_investment": initial_investment,
            "symbol": kwargs.get("symbol", "BRK-B"),
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

        # Structure equity curves with benchmark data
        x_dates, equity, bh_equity = structure_equity_curve(
            results["trade_dates"],
            results["portfolio_curve"],
            results["initial_investment"],
            None,  # Don't use BRK-B data for benchmark
            bh_symbol=symbol,
            benchmark_data=benchmark_data,
            standard_benchmark=standard_benchmark,
        )

        n_trades = 1  # Buy and hold = 1 trade
        avg_gain = np.mean(results["returns"]) * 100

        # Use standardized print function
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
    symbol = "BRK-B"
    years_back = 4
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)

    # Download benchmark data for consistency
    benchmark_data = download_data("SPY", start_date, end_date)

    strategy = BerkshireStrategy()
    results = strategy.run(data, 10000)

    # Add benchmark data to results
    results["benchmark_data"] = benchmark_data

    strategy.summarize(results, data)
