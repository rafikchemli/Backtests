import os
import numpy as np
import pandas as pd
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results
from core.plot_utils import plot_equity_curve

class OvernightSwingStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        trades, returns, trade_dates, portfolio_curve = self.backtest_overnight(data, initial_investment)
        return {
            'trades': trades,
            'returns': returns,
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': 'Overnight Swing',
            'initial_investment': initial_investment,
            'symbol': kwargs.get('symbol', 'SPY')
        }

    def backtest_overnight(self, data, initial_investment):
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]  # Start with the first date in the data
        for i in range(1, len(data)):
            entry = data['Close'].iloc[i-1]
            exit = data['Open'].iloc[i]
            trade_return = (exit - entry) / entry
            portfolio = portfolio * (1 + trade_return)
            trades.append((entry, exit, data.index[i]))
            portfolio_curve.append(portfolio)
            trade_dates.append(data.index[i])
        returns = [(exit-entry)/entry for entry, exit, _ in trades]
        return trades, returns, trade_dates, portfolio_curve

    def summarize(self, results, data=None):
        symbol = results.get('symbol', 'SPY')
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], results['portfolio_curve'], results['initial_investment'], data, bh_symbol=symbol)
        returns = np.diff(equity) / equity[:-1]
        n_trades = len(results['returns'])
        avg_gain = np.mean(results['returns']) * 100 if n_trades > 0 else 0
        
        # Use standardized print function
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "trade")
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'])

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "SPY"
    years_back = 30
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    strategy = OvernightSwingStrategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data) 