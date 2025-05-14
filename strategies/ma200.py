"""Strategy implementing the Moving Average 200 trading logic on financial data."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


class MA200Strategy(StrategyBase):
    """Strategy that trades based on the 200-day moving average crossover."""

    def run(self, data, initial_investment, **kwargs):
        """
        Run the MA200 strategy backtest.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters (e.g., symbol)

        Returns:
        - Dictionary with backtest results
        """
        # Calculate MA
        data = data.sort_index()
        data["MA200"] = data["Close"].rolling(window=200).mean()

        # Track trades and performance
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]

        # Initialize position state
        in_market = False
        entry_price = 0
        entry_date = None
        position_size = 0

        # Process data
        for i in range(200, len(data)):
            current_date = data.index[i]
            current_price = data["Close"].iloc[i]
            current_ma = data["MA200"].iloc[i]

            # Check for buy signal: price crosses above MA
            if not in_market and current_price > current_ma:
                entry_price = current_price
                entry_date = current_date
                position_size = portfolio / entry_price
                in_market = True
                trades.append((entry_price, None, entry_date, "BUY"))

            # Check for sell signal: price crosses below MA
            elif in_market and current_price < current_ma:
                exit_price = current_price
                portfolio = position_size * exit_price
                position_size = 0
                in_market = False

                # Update the last buy with this exit price
                for j in range(len(trades) - 1, -1, -1):
                    if trades[j][3] == "BUY" and trades[j][1] is None:
                        trades[j] = (trades[j][0], exit_price, trades[j][2], "BUY-SELL")
                        break

            # Update portfolio value
            if in_market:
                portfolio = position_size * current_price

            # Update tracking
            portfolio_curve.append(portfolio)
            trade_dates.append(current_date)

        # If still in market at end, close position
        if in_market:
            exit_price = data["Close"].iloc[-1]
            portfolio = position_size * exit_price
            portfolio_curve[-1] = portfolio

            # Update the last buy with this exit price
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
            "strategy_name": "MA 200",
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
            standard_benchmark=standard_benchmark,
        )

        # Calculate metrics
        n_trades = len([t for t in results["trades"] if t[3] in ["BUY", "BUY-SELL"]])
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
            "trade",
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

    strategy = MA200Strategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data)
