"""Strategy implementing the Monday Reversal trading logic on financial data."""

import numpy as np
import pandas as pd

from core.backtest_utils import download_data, print_results, structure_equity_curve
from core.plot_utils import plot_equity_curve
from core.strategy_base import StrategyBase


class MondayReversalStrategy(StrategyBase):
    """Strategy that buys on Mondays following a down day and sells on Mondays
    following an up day."""

    def run(self, data, initial_investment, **kwargs):
        """
        Run the Monday Reversal strategy backtest.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest
        - kwargs: Additional parameters (e.g., symbol)

        Returns:
        - Dictionary with backtest results
        """
        trades, returns, trade_dates, portfolio_curve = self.backtest_monday_reversal(
            data, initial_investment
        )
        return {
            "trades": trades,
            "returns": returns,
            "trade_dates": trade_dates,
            "portfolio_curve": portfolio_curve,
            "strategy_name": "Monday Reversal",
            "initial_investment": initial_investment,
            "symbol": kwargs.get("symbol", "SPY"),
        }

    def backtest_monday_reversal(self, data, initial_investment):
        """
        Implement Monday Reversal strategy logic.

        Buy on Mondays when the previous trading day closed down,
        sell on Mondays when the previous trading day closed up.

        Parameters:
        - data: Price data for the asset
        - initial_investment: Initial capital to invest

        Returns:
        - Tuple of (trades, returns, trade_dates, portfolio_curve)
        """
        # Track all trades for reporting
        trades = []
        # Get column names
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        open_col = "Open" if "Open" in data.columns else data.columns[0]

        # Sort data by date
        data = data.sort_index()

        # Initialize portfolio and tracking variables
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]

        # Initialize position state (True = in market, False = out of market)
        in_market = False  # Start not invested
        position_size = 0
        buy_trades = 0
        sell_trades = 0

        # Process the data day by day
        for i in range(1, len(data)):
            curr_date = data.index[i]
            curr_price = data[close_col].iloc[i]
            weekday = curr_date.weekday()

            # Monday check (weekday == 0)
            if weekday == 0:
                # Check if previous trading day closed down
                prev_idx = i - 1
                if prev_idx >= 0:
                    prev_day = data.index[prev_idx]
                    prev_open = data.loc[prev_day, open_col]
                    prev_close = data.loc[prev_day, close_col]

                    # Handle Series or similar data types
                    if isinstance(prev_open, (pd.Series, np.ndarray, list)):
                        prev_open = np.array(prev_open).item()
                    if isinstance(prev_close, (pd.Series, np.ndarray, list)):
                        prev_close = np.array(prev_close).item()

                    # Check if previous day closed down
                    prev_day_down = prev_close < prev_open

                    # Determine action based on market condition and current position
                    if prev_day_down:
                        # Previous day closed lower than it opened
                        if not in_market:
                            # Enter the market
                            entry_price = data.loc[curr_date, open_col]
                            if isinstance(entry_price, (pd.Series, np.ndarray, list)):
                                entry_price = np.array(entry_price).item()
                            position_size = portfolio / entry_price
                            in_market = True
                            trades.append((entry_price, None, curr_date, "BUY"))
                            buy_trades += 1
                    else:
                        # Previous day did not close down - signal to be out of market
                        if in_market:
                            # Exit the market
                            exit_price = data.loc[curr_date, open_col]
                            if isinstance(exit_price, (pd.Series, np.ndarray, list)):
                                exit_price = np.array(exit_price).item()
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
                            sell_trades += 1

            # Update portfolio value if in market
            if in_market:
                portfolio = position_size * curr_price

            # Update tracking variables
            portfolio_curve.append(portfolio)
            trade_dates.append(curr_date)

        # If we're still in the market at the end,
        # close the position using the last price
        if in_market:
            exit_price = data[close_col].iloc[-1]
            portfolio = position_size * exit_price
            # Update the last buy with this exit price
            for j in range(len(trades) - 1, -1, -1):
                if trades[j][3] == "BUY" and trades[j][1] is None:
                    trades[j] = (trades[j][0], exit_price, trades[j][2], "BUY-SELL")
                    break
            sell_trades += 1
            portfolio_curve[-1] = portfolio

        # Calculate returns for completed trades
        trade_returns = []
        for entry, exit, _, trade_type in trades:
            if entry is not None and exit is not None:
                trade_returns.append((exit - entry) / entry)

        return trades, trade_returns, trade_dates, portfolio_curve

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

        n_trades = len([t for t in results["trades"] if t[3] in ["BUY", "BUY-SELL"]])
        avg_gain = (
            np.mean(results["returns"]) * 100 if len(results["returns"]) > 0 else 0
        )

        # Use standardized print function
        print_results(
            results["strategy_name"], equity, bh_equity, n_trades, avg_gain, "trade"
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

    strategy = MondayReversalStrategy()
    results = strategy.run(data, 10000)

    # Add benchmark data to results
    results["benchmark_data"] = benchmark_data

    strategy.summarize(results, data)
