import os
import numpy as np
import pandas as pd
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results
from core.plot_utils import plot_equity_curve

class TurnOfMonthStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        boundaries = self.get_month_boundaries(data)
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = []
        for entry_date, exit_date in boundaries:
            entry = data.loc[entry_date, close_col]
            exit = data.loc[exit_date, close_col]
            trade_return = (exit - entry) / entry
            portfolio = portfolio * (1 + trade_return)
            trades.append((entry, exit, exit_date))
            portfolio_curve.append(portfolio)
            trade_dates.append(exit_date)
        returns = [(exit-entry)/entry for entry, exit, _ in trades]
        return {
            'trades': trades,
            'returns': returns,
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': 'Turn of the Month',
            'initial_investment': initial_investment,
            'symbol': kwargs.get('symbol', 'SPY')
        }

    def get_month_boundaries(self, data):
        month_groups = data.groupby([data.index.year, data.index.month])
        boundaries = []
        months = list(month_groups.groups.keys())
        for i in range(len(months)-1):
            this_month_idx = month_groups.groups[months[i]]
            next_month_idx = month_groups.groups[months[i+1]]
            if len(this_month_idx) < 5 or len(next_month_idx) < 3:
                continue
            fifth_last = this_month_idx[-5]
            third_new = next_month_idx[2]
            boundaries.append((fifth_last, third_new))
        return boundaries

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
    strategy = TurnOfMonthStrategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data) 