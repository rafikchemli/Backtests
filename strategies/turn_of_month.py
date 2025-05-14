import os
import numpy as np
import pandas as pd
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results, download_data
from core.plot_utils import plot_equity_curve

class TurnOfMonthStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        
        # Ensure we have the first date for proper benchmark comparison
        first_date = data.index[0]
        
        # Get month boundaries for trading
        boundaries = self.get_month_boundaries(data)
        trades = []
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [first_date]  # Start with the first date for alignment with other strategies
        
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
        # Use SPY as standard benchmark
        symbol = 'SPY'  # Always use SPY as standard benchmark
        
        # Get benchmark data (SPY) if available
        benchmark_data = results.get('benchmark_data', None)
        
        # Get standardized benchmark if available
        standard_benchmark = results.get('standard_benchmark', None)
        
        # If no benchmark data provided, download it (fallback)
        if benchmark_data is None:
            start_date = results['trade_dates'][0]
            end_date = results['trade_dates'][-1]
            benchmark_data = download_data(symbol, start_date, end_date)
        
        # Structure equity curves with standardized benchmark
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], 
            results['portfolio_curve'], 
            results['initial_investment'], 
            data, 
            bh_symbol=symbol,
            benchmark_data=benchmark_data,
            standard_benchmark=standard_benchmark
        )
        
        # Calculate metrics
        returns = np.diff(equity) / equity[:-1]
        n_trades = len(results['returns'])
        avg_gain = np.mean(results['returns']) * 100 if n_trades > 0 else 0
        
        # Use standardized print function with standard benchmark
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "trade", standard_benchmark)
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'])

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "SPY"
    years_back = 20  # Match the value in main.py
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    
    # Download benchmark data for consistency
    benchmark_data = download_data('SPY', start_date, end_date)
    
    # Create standardized benchmark
    standard_benchmark = {}
    close_col = 'Close' if 'Close' in benchmark_data.columns else benchmark_data.columns[0]
    start_price = benchmark_data[close_col].iloc[0]
    initial_investment = 10000
    
    for date in benchmark_data.index:
        price = benchmark_data.loc[date, close_col]
        equity = initial_investment * (price / start_price)
        standard_benchmark[date] = equity
    
    strategy = TurnOfMonthStrategy()
    results = strategy.run(data, 10000)
    
    # Add benchmark data to results
    results['benchmark_data'] = benchmark_data
    results['standard_benchmark'] = standard_benchmark
    
    strategy.summarize(results, data) 